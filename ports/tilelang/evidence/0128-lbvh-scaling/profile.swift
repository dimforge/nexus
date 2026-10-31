import Foundation
import Metal

struct Binding: Decodable { let entry: String; let buffers: [String]; let blocks: Int; let threads: Int; let iterations: Int }
struct Stage: Decodable { let name: String; let source: String; let bindings: Binding; let buffer_map: [String: String] }
struct Program: Decodable { let storage_buffers: [String]; let stages: [Stage]; let iterations: Int }
struct Counter: Decodable { let buffer: String; let word: Int }
struct Gate: Decodable { let id: String; let after_stage: String; let capacity: UInt64; let counters: [Counter] }
struct Frame: Decodable { let program: Program; let gates: [Gate] }
struct Metadata: Decodable { let libraries: [String: String] }
struct Prepared { let stage: Stage; let pipeline: MTLComputePipelineState }
struct Result { let wall: [Double]; let gpu: [Double]; let snapshots: [[Data]]; let submissions: Int }

func run() throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    func read(_ name: String) throws -> Data { try Data(contentsOf: root.appendingPathComponent(name)) }
    let decoder = JSONDecoder()
    let frame = try decoder.decode(Frame.self, from: read("frame.json"))
    let metadata = try decoder.decode(Metadata.self, from: read("specialization.json"))
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fatalError("No Metal device") }
    precondition(frame.program.iterations == 1)
    var pipelines: [String: MTLComputePipelineState] = [:]
    var prepared: [Prepared] = []
    for stage in frame.program.stages {
        let filename = metadata.libraries[stage.source]!
        let key = filename + ":" + stage.bindings.entry
        if pipelines[key] == nil {
            let library = try device.makeLibrary(URL: root.appendingPathComponent(filename))
            let function = library.makeFunction(name: stage.bindings.entry)!
            pipelines[key] = try device.makeComputePipelineState(function: function)
        }
        prepared.append(Prepared(stage: stage, pipeline: pipelines[key]!))
    }
    var gates: [Int: [Gate]] = [:]
    for gate in frame.gates {
        let end = prepared.firstIndex(where: { $0.stage.name == gate.after_stage })! + 1
        gates[end, default: []].append(gate)
    }
    let ends = Set(gates.keys).union([prepared.count]).sorted()
    let initial = try frame.program.storage_buffers.map { try read($0 + ".bin") }


    var buffers: [String: MTLBuffer] = [:]
    for (name, bytes) in zip(frame.program.storage_buffers, initial) {
        buffers[name] = bytes.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: bytes.count, options: .storageModeShared)! }
    }
    let resolved = prepared.map { p in p.stage.bindings.buffers.map { buffers[p.stage.buffer_map[$0]!]! } }
    var records: [[String: Any]] = []
    for step in 0..<2 {
        var total = 0.0
        for (index, p) in prepared.enumerated() {
            let ms: Double = try autoreleasepool {
                let command = queue.makeCommandBuffer()!
                for _ in 0..<p.stage.bindings.iterations {
                    let encoder = command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(p.pipeline)
                    for (slot, buffer) in resolved[index].enumerated() { encoder.setBuffer(buffer, offset: 0, index: slot) }
                    encoder.dispatchThreadgroups(MTLSize(width: p.stage.bindings.blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: p.stage.bindings.threads, height: 1, depth: 1))
                    encoder.endEncoding()
                }
                command.commit(); command.waitUntilCompleted()
                if let error = command.error { throw error }
                return 1000 * max(0, command.gpuEndTime - command.gpuStartTime)
            }
            total += ms
            records.append(["frame": step, "stage": p.stage.name, "gpu_ms": ms, "blocks": p.stage.bindings.blocks, "threads": p.stage.bindings.threads])
            for gate in gates[index+1] ?? [] {
                let required = gate.counters.reduce(UInt64(0)) { total, counter in
                    total + UInt64(buffers[counter.buffer]!.contents().assumingMemoryBound(to: UInt32.self)[counter.word])
                }
                guard required <= gate.capacity else { throw NSError(domain: "Capacity", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(gate.id): \(required) > \(gate.capacity)"]) }
            }
        }
        fputs("frame \(step): \(total) ms GPU\n", stderr)
    }
    let output: [String: Any] = ["device": device.name, "stages": prepared.count, "buffer_bytes": initial.reduce(0) { $0 + $1.count }, "records": records, "scope": "Two-frame per-stage GPU timestamps, saved compiled Balls scene. Serial submissions for attribution, not production latency."]
    let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}
do { try run() } catch { fputs("Probe failed: \(error)\n", stderr); exit(1) }
