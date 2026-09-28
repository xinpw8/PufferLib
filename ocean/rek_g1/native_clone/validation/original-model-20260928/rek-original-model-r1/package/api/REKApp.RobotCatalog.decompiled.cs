using System;
using System.Runtime.CompilerServices;
using Il2CppInterop.Common.Attributes;
using Il2CppInterop.Runtime;
using Il2CppInterop.Runtime.InteropTypes;
using Il2CppInterop.Runtime.Runtime;
using Il2CppSystem.Collections.Generic;
using UnityEngine;

namespace REKApp;

public class RobotCatalog : ScriptableObject
{
	private static readonly IntPtr NativeFieldInfoPtr_robots;

	private static readonly IntPtr NativeMethodInfoPtr_get_Robots_Public_get_IReadOnlyList_1_RobotCatalogEntry_0;

	private static readonly IntPtr NativeMethodInfoPtr_TryGetById_Public_Boolean_String_byref_RobotCatalogEntry_0;

	private static readonly IntPtr NativeMethodInfoPtr__ctor_Public_Void_0;

	public unsafe List<RobotCatalogEntry> robots
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_robots);
			IntPtr intPtr = *(IntPtr*)num;
			return (intPtr != (IntPtr)0) ? Il2CppObjectPool.Get<List<RobotCatalogEntry>>(intPtr) : null;
		}
		set
		{
			IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_robots)), IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)list));
		}
	}

	public unsafe IReadOnlyList<RobotCatalogEntry> Robots
	{
		[CallerCount(24)]
		[CachedScanResults(RefRangeStart = 17903, RefRangeEnd = 17927, XrefRangeStart = 17903, XrefRangeEnd = 17927, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
		get
		{
			IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IntPtr* ptr = null;
			Unsafe.SkipInit(out IntPtr intPtr2);
			IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr_get_Robots_Public_get_IReadOnlyList_1_RobotCatalogEntry_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
			Il2CppException.RaiseExceptionIfNecessary(intPtr2);
			return (intPtr != (IntPtr)0) ? Il2CppObjectPool.Get<IReadOnlyList<RobotCatalogEntry>>(intPtr) : null;
		}
	}

	static RobotCatalog()
	{
		Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr = IL2CPP.GetIl2CppClass("REKApp.dll", "REKApp", "RobotCatalog");
		IL2CPP.il2cpp_runtime_class_init(Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr);
		NativeFieldInfoPtr_robots = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr, "robots");
		NativeMethodInfoPtr_get_Robots_Public_get_IReadOnlyList_1_RobotCatalogEntry_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr, 100669708);
		NativeMethodInfoPtr_TryGetById_Public_Boolean_String_byref_RobotCatalogEntry_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr, 100669709);
		NativeMethodInfoPtr__ctor_Public_Void_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr, 100669710);
	}

	[CallerCount(10)]
	[CachedScanResults(RefRangeStart = 357144, RefRangeEnd = 357154, XrefRangeStart = 357134, XrefRangeEnd = 357144, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
	public unsafe bool TryGetById(string id, out RobotCatalogEntry entry)
	{
		IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
		IntPtr* ptr = stackalloc IntPtr[2];
		*ptr = IL2CPP.ManagedStringToIl2Cpp(id);
		byte* num = (byte*)ptr + checked((nuint)1u * unchecked((nuint)sizeof(IntPtr)));
		nint num2 = 0;
		*(nint**)num = &num2;
		Unsafe.SkipInit(out IntPtr intPtr2);
		IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr_TryGetById_Public_Boolean_String_byref_RobotCatalogEntry_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
		Il2CppException.RaiseExceptionIfNecessary(intPtr2);
		nint num3 = num2;
		entry = ((num3 == 0) ? null : new RobotCatalogEntry(num3));
		return *(bool*)IL2CPP.il2cpp_object_unbox(intPtr);
	}

	[CallerCount(0)]
	[CachedScanResults(RefRangeStart = 0, RefRangeEnd = 0, XrefRangeStart = 357154, XrefRangeEnd = 357161, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
	public unsafe RobotCatalog()
		: this(IL2CPP.il2cpp_object_new(Il2CppClassPointerStore<RobotCatalog>.NativeClassPtr))
	{
		IntPtr* ptr = null;
		Unsafe.SkipInit(out IntPtr intPtr2);
		IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr__ctor_Public_Void_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
		Il2CppException.RaiseExceptionIfNecessary(intPtr2);
	}

	public RobotCatalog(IntPtr pointer)
		: base(pointer)
	{
	}
}
