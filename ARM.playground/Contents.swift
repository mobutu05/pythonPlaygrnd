import Cocoa

var instr:uint32 = 0xFFEEAABB

let cond = instr >> 28

print("\(cond)")
enum ProcessorMode{
    case User
}
class Arm{
    var R0:uint32 = 0
    var R15:uint32 = 0
}

