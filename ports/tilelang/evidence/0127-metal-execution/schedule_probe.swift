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

    func replay(_ batched: Bool) throws -> Result {
        var buffers: [String: MTLBuffer] = [:]
        for (name, bytes) in zip(frame.program.storage_buffers, initial) {
            buffers[name] = bytes.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: bytes.count, options: .storageModeShared)! }
        }
        let resolved = prepared.map { p in p.stage.bindings.buffers.map { buffers[p.stage.buffer_map[$0]!]! } }
        var wall: [Double] = [], gpu: [Double] = [], snapshots: [[Data]] = []
        var submissions = 0
        func submit(_ command: MTLCommandBuffer) throws -> Double {
            submissions += 1
            command.commit(); command.waitUntilCompleted()
            if let error = command.error { throw error }
            return max(0, command.gpuEndTime - command.gpuStartTime)
        }
        func encode(_ index: Int, _ command: MTLCommandBuffer) {
            let p = prepared[index]
            let encoder = command.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(p.pipeline)
            for (slot, buffer) in resolved[index].enumerated() { encoder.setBuffer(buffer, offset: 0, index: slot) }
            encoder.dispatchThreadgroups(MTLSize(width: p.stage.bindings.blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: p.stage.bindings.threads, height: 1, depth: 1))
            encoder.endEncoding()
        }
        for _ in 0..<12 {
            let measurement: (Double, Double) = try autoreleasepool {
                let startTime = ProcessInfo.processInfo.systemUptime
                var deviceSeconds = 0.0
                var start = 0
                for end in ends {
                    if batched {
                        let command = queue.makeCommandBuffer()!
                        for index in start..<end {
                            for _ in 0..<prepared[index].stage.bindings.iterations { encode(index, command) }
                        }
                        deviceSeconds += try submit(command)
                    } else {
                        for index in start..<end {
                            // Match khal::Metal::synchronize before every stage.
                            _ = try submit(queue.makeCommandBuffer()!)
                            for _ in 0..<prepared[index].stage.bindings.iterations {
                                let command = queue.makeCommandBuffer()!
                                encode(index, command)
                                deviceSeconds += try submit(command)
                            }
                        }
                    }
                    for gate in gates[end] ?? [] {
                        let required = gate.counters.reduce(UInt64(0)) { total, counter in
                            total + UInt64(buffers[counter.buffer]!.contents().assumingMemoryBound(to: UInt32.self)[counter.word])
                        }
                        guard required <= gate.capacity else { throw NSError(domain: "Capacity", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(gate.id): \(required) > \(gate.capacity)"]) }
                    }
                    start = end
                }
                return (ProcessInfo.processInfo.systemUptime - startTime, deviceSeconds)
            }
            wall.append(measurement.0 * 1000)
            gpu.append(measurement.1 * 1000)
            // Diagnostic state capture is outside the timed region in both modes.
            snapshots.append(frame.program.storage_buffers.map { name in
                let b = buffers[name]!
                return Data(bytes: b.contents(), count: b.length)
            })
        }
        return Result(wall: wall, gpu: gpu, snapshots: snapshots, submissions: submissions / 12)
    }
    func median(_ values: [Double]) -> Double { let v = values.sorted(); return (v[(v.count-1)/2]+v[v.count/2])/2 }
    let order = [false, true, true, false]
    var reference: [[Data]]? = nil
    var records: [[String: Any]] = []
    var disagreements: [String] = []
    for (round, batched) in order.enumerated() {
        let result = try replay(batched)
        if let reference {
            for f in 0..<result.snapshots.count {
                for b in 0..<result.snapshots[f].count where result.snapshots[f][b] != reference[f][b] {
                    disagreements.append("run \(round), frame \(f), buffer \(frame.program.storage_buffers[b])")
                }
            }
        } else { reference = result.snapshots }
        records.append(["mode": batched ? "batched_at_capacity_gates" : "per_stage_flush_and_wait", "submissions_per_frame": result.submissions, "median_wall_ms_excluding_first_two": median(Array(result.wall.dropFirst(2))), "median_gpu_ms_excluding_first_two": median(Array(result.gpu.dropFirst(2))), "wall_ms": result.wall, "gpu_ms": result.gpu])
    }
    if CommandLine.arguments.count > 2, let reference {
        let dump = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
        for (frameIndex, snapshot) in reference.enumerated() {
            for (name, data) in zip(frame.program.storage_buffers, snapshot) {
                try data.write(to: dump.appendingPathComponent("\(frameIndex)-\(name).bin"))
            }
        }
    }
    let output: [String: Any] = ["device": device.name, "stages": prepared.count, "frames_per_run": 12, "initial_buffers": initial.count, "runs": records, "all_frame_buffers_bitwise_equal": disagreements.isEmpty, "disagreements": disagreements, "scope": "Temporary Swift replay of the exact already-compiled Metal libraries. Only submission strategy differs. Capacity gates retained. Not the production runtime, a Rust-GPU comparison, or full parity qualification."]
    let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}
do { try run() } catch { fputs("Probe failed: \(error)\n", stderr); exit(1) }
