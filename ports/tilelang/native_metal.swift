// Replay generated MSL directly. Python/TVM/PyTorch are not used by this process.
import Foundation
import Metal

struct Manifest: Decodable {
    let entry: String
    let buffers: [String]
    let storage_buffers: [String]
    let blocks: Int
    let threads: Int
    let iterations: Int
}
func runReplay() throws {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    func read(_ name: String) throws -> Data { try Data(contentsOf: root.appendingPathComponent(name)) }
    let manifest = try JSONDecoder().decode(Manifest.self, from: read("bindings.json"))
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        fatalError("No Metal device/queue")
    }
    let source = String(data: try read("kernel.metal"), encoding: .utf8)!
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(source: source, options: options)
    guard let function = library.makeFunction(name: manifest.entry) else { fatalError("Missing entry point") }
    let pipeline = try device.makeComputePipelineState(function: function)
    precondition(manifest.threads <= pipeline.maxTotalThreadsPerThreadgroup)
    var buffers: [String: MTLBuffer] = [:]
    for name in manifest.storage_buffers {
        let data = try read(name + ".bin")
        precondition(!data.isEmpty && data.count % 4 == 0)
        buffers[name] = data.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)!
        }
    }
    // Separate ordered dispatches. No cross-workgroup synchronization is assumed.
    for _ in 0..<manifest.iterations {
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for (slot, name) in manifest.buffers.enumerated() {
            encoder.setBuffer(buffers[name], offset: 0, index: slot)
        }
        encoder.dispatchThreadgroups(MTLSize(width: manifest.blocks, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: manifest.threads, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
    }
    for name in manifest.storage_buffers {
        let buffer = buffers[name]!
        try Data(bytes: buffer.contents(), count: buffer.length).write(to: root.appendingPathComponent(name + ".actual.bin"))
    }
    let metadata: [String: Any] = ["device": device.name, "runtime": "native Metal", "os": ProcessInfo.processInfo.operatingSystemVersionString]
    let json = try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])
    print(String(data: json, encoding: .utf8)!)
}

do {
    try runReplay()
} catch {
    fputs("Metal replay failed: \(error)\n", stderr)
    exit(1)
}
