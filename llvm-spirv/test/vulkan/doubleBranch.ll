; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --vulkan --spirv-ext=+all %t.bc -o %t.spv
; RUNx: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val --target-env vulkan1.1 %t.spv
; ModuleID = 'mainVO3.bc'
; ModuleID = 'mainVO3.bc'
source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-vulkan-windows-sycldevice"

%struct._ZTS8_arg_0_t._arg_0_t = type { i32 addrspace(9)* }
%struct._ZTS8_arg_3_t._arg_3_t = type { %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }

$_ZTSN2cl4sycl6kernelE = comdat any

@_arg_0 = external dso_local local_unnamed_addr addrspace(9) global %struct._ZTS8_arg_0_t._arg_0_t, align 8
@_arg_3 = external dso_local local_unnamed_addr addrspace(9) global %struct._ZTS8_arg_3_t._arg_3_t, align 8
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(7) constant <3 x i32>, align 16

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSN2cl4sycl6kernelE() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 {
entry:
  %0 = load i32 addrspace(9)*, i32 addrspace(9)* addrspace(9)* getelementptr inbounds (%struct._ZTS8_arg_0_t._arg_0_t, %struct._ZTS8_arg_0_t._arg_0_t addrspace(9)* @_arg_0, i64 0, i32 0), align 8, !tbaa !6
  %agg.tmp2.sroa.0.0.copyload = load i64, i64 addrspace(9)* getelementptr inbounds (%struct._ZTS8_arg_3_t._arg_3_t, %struct._ZTS8_arg_3_t._arg_3_t addrspace(9)* @_arg_3, i64 0, i32 0, i32 0, i32 0, i64 0), align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(9)* %0, i64 %agg.tmp2.sroa.0.0.copyload
  %1 = load <3 x i32>, <3 x i32> addrspace(7)* @__spirv_BuiltInGlobalInvocationId, align 16, !noalias !11
  %2 = extractelement <3 x i32> %1, i64 0
  %conv.i.i.i.i.i.i = zext i32 %2 to i64
  %rem.i = and i64 %conv.i.i.i.i.i.i, 1
  %cmp.i = icmp eq i64 %rem.i, 0
  %cmp.i.inv = xor i1 %cmp.i, true
  br i1 %cmp.i.inv, label %"entry._ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit_crit_edge", label %Flow2

"entry._ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit_crit_edge": ; preds = %entry
  %ptridx.i.i.phi.trans.insert = getelementptr inbounds i32, i32 addrspace(9)* %add.ptr.i, i64 %conv.i.i.i.i.i.i
  %.pre = load i32, i32 addrspace(9)* %ptridx.i.i.phi.trans.insert, align 4, !tbaa !20
  br label %Flow2

Flow2:                                            ; preds = %"entry._ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit_crit_edge", %entry
  %3 = phi i32 [ %.pre, %"entry._ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit_crit_edge" ], [ undef, %entry ]
  %4 = phi i1 [ false, %"entry._ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit_crit_edge" ], [ true, %entry ]
  br i1 %4, label %if.then.i, label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit"

if.then.i:                                        ; preds = %Flow2
  %rem3.i = and i64 %conv.i.i.i.i.i.i, 3
  %cmp4.i = icmp eq i64 %rem3.i, 0
  %cmp4.i.inv = xor i1 %cmp4.i, true
  %ptridx.i38.i = getelementptr inbounds i32, i32 addrspace(9)* %add.ptr.i, i64 %conv.i.i.i.i.i.i
  br i1 %cmp4.i.inv, label %if.else.i, label %Flow

Flow:                                             ; preds = %if.else.i, %if.then.i
  %5 = phi i32 [ %7, %if.else.i ], [ undef, %if.then.i ]
  %6 = phi i1 [ false, %if.else.i ], [ true, %if.then.i ]
  br i1 %6, label %if.then5.i, label %Flow1

if.then5.i:                                       ; preds = %Flow
  store i32 4, i32 addrspace(9)* %ptridx.i38.i, align 4, !tbaa !20
  br label %Flow1

if.else.i:                                        ; preds = %if.then.i
  %7 = load i32, i32 addrspace(9)* %ptridx.i38.i, align 4, !tbaa !20
  %conv10.i = mul i32 %7, %2
  %add.i = add nuw nsw i64 %conv.i.i.i.i.i.i, 1
  %ptridx.i23.i = getelementptr inbounds i32, i32 addrspace(9)* %add.ptr.i, i64 %add.i
  store i32 %conv10.i, i32 addrspace(9)* %ptridx.i23.i, align 4, !tbaa !20
  br label %Flow

Flow1:                                            ; preds = %if.then5.i, %Flow
  %8 = phi i32 [ 4, %if.then5.i ], [ %5, %Flow ]
  br label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit"

"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_4itemILi1ELb1EEEE_clES5_.exit": ; preds = %Flow2, %Flow1
  %9 = phi i32 [ %3, %Flow2 ], [ %8, %Flow1 ]
  %ptridx.i.i = getelementptr inbounds i32, i32 addrspace(9)* %add.ptr.i, i64 %conv.i.i.i.i.i.i
  %add17.i = add nsw i32 %9, 2
  store i32 %add17.i, i32 addrspace(9)* %ptridx.i.i, align 4, !tbaa !20
  ret void
}

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="main.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 12.0.0 (git@github.com:tadeaustria/llvm.git eb8c30756a0d43c755c5ffaeec029cc77142baf8)"}
!5 = !{}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS8_arg_0_t", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !14, !16, !18}
!12 = distinct !{!12, !13, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!13 = distinct !{!13, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!14 = distinct !{!14, !15, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!15 = distinct !{!15, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!16 = distinct !{!16, !17, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!17 = distinct !{!17, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!18 = distinct !{!18, !19, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!19 = distinct !{!19, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !9, i64 0}