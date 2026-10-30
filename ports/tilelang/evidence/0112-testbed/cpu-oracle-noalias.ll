; ModuleID = 'noalias_probe.4284eef7bc0d6a61-cgu.0'
source_filename = "noalias_probe.4284eef7bc0d6a61-cgu.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx11.0.0"

; Function Attrs: uwtable
define noundef i32 @mutable_reference_probe(ptr nofree noundef align 4 dereferenceable(4) %x, ptr nofree noundef align 4 dereferenceable(4) %y) unnamed_addr #0 {
start:
  store i32 1, ptr %x, align 4
  store i32 2, ptr %y, align 4
  %_0 = load i32, ptr %x, align 4, !noundef !4
  ret i32 %_0
}

; Function Attrs: uwtable
define noundef i32 @shared_reference_probe(ptr noalias nofree noundef readonly align 4 captures(address, read_provenance) dereferenceable(4) %x) unnamed_addr #0 {
start:
  %_0 = load i32, ptr %x, align 4, !noundef !4
  ret i32 %_0
}

attributes #0 = { uwtable "frame-pointer"="non-leaf" "probe-stack"="inline-asm" "target-cpu"="apple-m1" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 1}
!3 = !{!"rustc version 1.98.1 (48a229cea 2026-09-01)"}
!4 = !{}
