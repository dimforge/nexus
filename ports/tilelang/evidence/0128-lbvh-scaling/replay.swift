import Foundation
import Metal
import CryptoKit

struct Binding: Decodable { let entry: String; let buffers: [String]; let blocks: Int; let threads: Int; let iterations: Int }
struct Stage: Decodable { let name: String; let source: String; let bindings: Binding; let buffer_map: [String: String] }
struct Program: Decodable { let storage_buffers: [String]; let stages: [Stage]; let iterations: Int }
struct Counter: Decodable { let buffer: String; let word: Int }
struct Gate: Decodable { let id: String; let after_stage: String; let capacity: UInt64; let counters: [Counter] }
struct Frame: Decodable { let program: Program; let gates: [Gate] }
struct Metadata: Decodable { let libraries: [String: String] }
struct Prepared { let stage: Stage; let pipeline: MTLComputePipelineState }
struct Result { let wall: [Double]; let gpu: [Double]; let snapshots: [[String: String]]; let submissions: Int }

func run() throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    func read(_ name: String) throws -> Data { try Data(contentsOf: root.appendingPathComponent(name)) }
    let decoder = JSONDecoder()
    let frame = try decoder.decode(Frame.self, from: read("frame.json"))
    let metadata = try decoder.decode(Metadata.self, from: read("specialization.json"))
    let layout = try JSONSerialization.jsonObject(with: read("layout.json")) as! [String: Any]
    let link = layout["link"] as! [String: Int]
    let linkWords = layout["link_words"] as! Int
    let tileWords = layout["tile_words"] as! Int
    let warmstart = layout["warmstart"] as! Int
    let cacheWords = 2 * (layout["velocity_words"] as! Int)
    let scene = try JSONSerialization.jsonObject(with: read("case.json")) as! [String: Any]
    let contactCapacity = scene["contact_capacity"] as! Int
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
        var wall: [Double] = [], gpu: [Double] = [], snapshots: [[String: String]] = []
        var submissions = 0
        func words(_ name: String) -> UnsafeBufferPointer<UInt32> {
            let b = buffers[name]!
            return UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: UInt32.self), count: b.length / 4)
        }
        var previousIndices = Array(words("OldIndices"))
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
        for step in 0..<6 {
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
            let names = ["Poses", "Velocities", "Contacts", "OldLinks", "OldTiles", "States", "OldColors", "OldIds", "OldCounts", "OldIndices", "LbvhTree", "LbvhDomain", "LbvhKeys", "LbvhSorted", "PairCount", "Stats"]
            func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
            var state: [String: String] = [:]
            for name in names {
                let b = buffers[name]!
                state[name] = hash(Data(bytesNoCopy: b.contents(), count: b.length, deallocator: .none))
            }
            let count = Int(buffers["PairCount"]!.contents().assumingMemoryBound(to: UInt32.self)[0])
            let pairs = buffers["Pairs"]!.contents().assumingMemoryBound(to: UInt32.self)
            let sortedPairs = (0..<count).map { UInt64(pairs[2*$0]) << 32 | UInt64(pairs[2*$0+1]) }.sorted()
            state["pair_set"] = sortedPairs.withUnsafeBytes { hash(Data($0)) }
            state["pair_count"] = String(count)
            // Bucket scatter legally allocates solver slots atomically. Compare
            // live tiles in canonical contact order, validating the old-slot link
            // before normalizing that one allocation-dependent field.
            let links = words("OldLinks"), indices = words("OldIndices"), tiles = words("OldTiles")
            var canonicalTiles: [UInt32] = [], canonicalLinks: [UInt32] = []
            for i in 0..<contactCapacity {
                let off = i * linkWords
                if links[off + link["len"]!] == 0 { continue }
                let previous = links[off + link["previous_constraint"]!]
                let oldSlot = links[off + link["previous_constraint_index"]!]
                precondition(oldSlot == (previous == UInt32.max ? UInt32.max : previousIndices[Int(previous)]))
                var row = Array(links[off..<(off+linkWords)])
                row[link["previous_constraint_index"]!] = 0
                canonicalLinks.append(UInt32(i)); canonicalLinks.append(contentsOf: row)
                let slot = Int(indices[i])
                let base = slot / 32 * tileWords + slot % 32 * 4
                for channel in 0..<(warmstart / 128) {
                    canonicalTiles.append(contentsOf: tiles[(base+channel*128)..<(base+channel*128+4)])
                }
                // The warmstart tail is AoS, unlike the tile's SoA fields.
                let cache = slot / 32 * tileWords + warmstart + slot % 32 * cacheWords
                canonicalTiles.append(contentsOf: tiles[cache..<(cache+cacheWords)])
            }
            if CommandLine.arguments.count > 2 {
                let directory = URL(fileURLWithPath: CommandLine.arguments[2])
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try canonicalTiles.withUnsafeBytes { try Data($0).write(to: directory.appendingPathComponent("tiles-\(step).bin")) }
                try canonicalLinks.withUnsafeBytes { try Data($0).write(to: directory.appendingPathComponent("links-\(step).bin")) }
            }
            state["canonical_tiles"] = canonicalTiles.withUnsafeBytes { hash(Data($0)) }
            state["canonical_links"] = canonicalLinks.withUnsafeBytes { hash(Data($0)) }
            previousIndices = Array(indices)
            snapshots.append(state)
        }
        return Result(wall: wall, gpu: gpu, snapshots: snapshots, submissions: submissions / 6)
    }
    func median(_ values: [Double]) -> Double { let v = values.sorted(); return (v[(v.count-1)/2]+v[v.count/2])/2 }
    let result = try replay(true)
    let output: [String: Any] = ["device": device.name, "stages": prepared.count, "frames_per_run": 6, "submissions_per_frame": result.submissions, "median_wall_ms_excluding_first_two": median(Array(result.wall.dropFirst(2))), "median_gpu_ms_excluding_first_two": median(Array(result.gpu.dropFirst(2))), "wall_ms": result.wall, "gpu_ms": result.gpu, "state_sha256": result.snapshots, "scope": "Six frames of exact compiled Metal libraries batched at capacity gates; hashes outside timing; excludes viewer publication and Rust runtime bookkeeping."]
    let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}
do { try run() } catch { fputs("Probe failed: \(error)\n", stderr); exit(1) }
