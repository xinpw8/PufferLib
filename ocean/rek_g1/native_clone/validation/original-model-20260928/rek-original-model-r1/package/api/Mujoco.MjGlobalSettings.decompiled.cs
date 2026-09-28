using System;
using System.Runtime.CompilerServices;
using Il2CppInterop.Common.Attributes;
using Il2CppInterop.Runtime;
using Il2CppInterop.Runtime.InteropTypes;
using Il2CppInterop.Runtime.Runtime;
using Il2CppSystem.Collections.Generic;
using Il2CppSystem.Xml;
using UnityEngine;

namespace Mujoco;

public class MjGlobalSettings : MonoBehaviour
{
	private static readonly IntPtr NativeFieldInfoPtr_DebugFileName;

	private static readonly IntPtr NativeFieldInfoPtr_MouseSpringStiffness;

	private static readonly IntPtr NativeFieldInfoPtr_UseRawGameObjectNames;

	private static readonly IntPtr NativeFieldInfoPtr_GlobalOptions;

	private static readonly IntPtr NativeFieldInfoPtr_GlobalSizes;

	private static readonly IntPtr NativeFieldInfoPtr_CustomNumeric;

	private static readonly IntPtr NativeFieldInfoPtr__instance;

	private static readonly IntPtr NativeMethodInfoPtr_get_Instance_Public_Static_get_MjGlobalSettings_0;

	private static readonly IntPtr NativeMethodInfoPtr_Awake_Public_Void_0;

	private static readonly IntPtr NativeMethodInfoPtr_ParseGlobalMjcfSections_Public_Void_XmlElement_0;

	private static readonly IntPtr NativeMethodInfoPtr_GlobalsToMjcf_Public_Void_XmlElement_0;

	private static readonly IntPtr NativeMethodInfoPtr__ctor_Public_Void_0;

	public unsafe string DebugFileName
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_DebugFileName);
			return IL2CPP.Il2CppStringToManaged(*(IntPtr*)num);
		}
		set
		{
			IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_DebugFileName)), IL2CPP.ManagedStringToIl2Cpp(text));
		}
	}

	public unsafe float MouseSpringStiffness
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_MouseSpringStiffness);
			return *(float*)num;
		}
		set
		{
			*(float*)((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_MouseSpringStiffness)) = num;
		}
	}

	public unsafe bool UseRawGameObjectNames
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_UseRawGameObjectNames);
			return *(bool*)num;
		}
		set
		{
			*(bool*)((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_UseRawGameObjectNames)) = flag;
		}
	}

	public unsafe MjOptionStruct GlobalOptions
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_GlobalOptions);
			return *(MjOptionStruct*)num;
		}
		set
		{
			*(MjOptionStruct*)((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_GlobalOptions)) = mjOptionStruct;
		}
	}

	public unsafe MjSizeStruct GlobalSizes
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_GlobalSizes);
			return new MjSizeStruct(IL2CPP.il2cpp_value_box(Il2CppClassPointerStore<MjSizeStruct>.NativeClassPtr, (IntPtr)num));
		}
		set
		{
			// IL cpblk instruction
			Unsafe.CopyBlock((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_GlobalSizes), IL2CPP.il2cpp_object_unbox(IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)mjSizeStruct)), IL2CPP.il2cpp_class_value_size(Il2CppClassPointerStore<MjSizeStruct>.NativeClassPtr, ref *(uint*)null));
		}
	}

	public unsafe List<NumericEntry> CustomNumeric
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_CustomNumeric);
			IntPtr intPtr = *(IntPtr*)num;
			return (intPtr != (IntPtr)0) ? Il2CppObjectPool.Get<List<NumericEntry>>(intPtr) : null;
		}
		set
		{
			IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_CustomNumeric)), IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)list));
		}
	}

	public unsafe static MjGlobalSettings _instance
	{
		get
		{
			Unsafe.SkipInit(out IntPtr intPtr);
			IL2CPP.il2cpp_field_static_get_value(NativeFieldInfoPtr__instance, (void*)(&intPtr));
			IntPtr intPtr2 = intPtr;
			return (intPtr2 != (IntPtr)0) ? Il2CppObjectPool.Get<MjGlobalSettings>(intPtr2) : null;
		}
		set
		{
			IL2CPP.il2cpp_field_static_set_value(NativeFieldInfoPtr__instance, (void*)IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)mjGlobalSettings));
		}
	}

	public unsafe static MjGlobalSettings Instance
	{
		[CallerCount(2)]
		[CachedScanResults(RefRangeStart = 232156, RefRangeEnd = 232158, XrefRangeStart = 232143, XrefRangeEnd = 232156, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
		get
		{
			IntPtr* ptr = null;
			Unsafe.SkipInit(out IntPtr intPtr2);
			IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr_get_Instance_Public_Static_get_MjGlobalSettings_0, (IntPtr)0, (void**)ptr, ref intPtr2);
			Il2CppException.RaiseExceptionIfNecessary(intPtr2);
			return (intPtr != (IntPtr)0) ? Il2CppObjectPool.Get<MjGlobalSettings>(intPtr) : null;
		}
	}

	static MjGlobalSettings()
	{
		Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr = IL2CPP.GetIl2CppClass("Mujoco.Runtime.dll", "Mujoco", "MjGlobalSettings");
		IL2CPP.il2cpp_runtime_class_init(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr);
		NativeFieldInfoPtr_DebugFileName = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "DebugFileName");
		NativeFieldInfoPtr_MouseSpringStiffness = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "MouseSpringStiffness");
		NativeFieldInfoPtr_UseRawGameObjectNames = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "UseRawGameObjectNames");
		NativeFieldInfoPtr_GlobalOptions = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "GlobalOptions");
		NativeFieldInfoPtr_GlobalSizes = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "GlobalSizes");
		NativeFieldInfoPtr_CustomNumeric = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "CustomNumeric");
		NativeFieldInfoPtr__instance = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, "_instance");
		NativeMethodInfoPtr_get_Instance_Public_Static_get_MjGlobalSettings_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, 100663808);
		NativeMethodInfoPtr_Awake_Public_Void_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, 100663809);
		NativeMethodInfoPtr_ParseGlobalMjcfSections_Public_Void_XmlElement_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, 100663810);
		NativeMethodInfoPtr_GlobalsToMjcf_Public_Void_XmlElement_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, 100663811);
		NativeMethodInfoPtr__ctor_Public_Void_0 = IL2CPP.GetIl2CppMethodByToken(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr, 100663812);
	}

	[CallerCount(0)]
	[CachedScanResults(RefRangeStart = 0, RefRangeEnd = 0, XrefRangeStart = 232158, XrefRangeEnd = 232169, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
	public unsafe void Awake()
	{
		IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
		IntPtr* ptr = null;
		Unsafe.SkipInit(out IntPtr intPtr2);
		IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr_Awake_Public_Void_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
		Il2CppException.RaiseExceptionIfNecessary(intPtr2);
	}

	[CallerCount(1)]
	[CachedScanResults(RefRangeStart = 232229, RefRangeEnd = 232230, XrefRangeStart = 232169, XrefRangeEnd = 232229, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
	public unsafe void ParseGlobalMjcfSections(XmlElement mujocoNode)
	{
		IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
		IntPtr* ptr = stackalloc IntPtr[1];
		*ptr = IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)mujocoNode);
		Unsafe.SkipInit(out IntPtr intPtr2);
		IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr_ParseGlobalMjcfSections_Public_Void_XmlElement_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
		Il2CppException.RaiseExceptionIfNecessary(intPtr2);
	}

	[CallerCount(1)]
	[CachedScanResults(RefRangeStart = 232276, RefRangeEnd = 232277, XrefRangeStart = 232230, XrefRangeEnd = 232276, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
	public unsafe void GlobalsToMjcf(XmlElement mjcf)
	{
		IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
		IntPtr* ptr = stackalloc IntPtr[1];
		*ptr = IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)mjcf);
		Unsafe.SkipInit(out IntPtr intPtr2);
		IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr_GlobalsToMjcf_Public_Void_XmlElement_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
		Il2CppException.RaiseExceptionIfNecessary(intPtr2);
	}

	[CallerCount(0)]
	[CachedScanResults(RefRangeStart = 0, RefRangeEnd = 0, XrefRangeStart = 232277, XrefRangeEnd = 232292, MetadataInitTokenRva = 0L, MetadataInitFlagRva = 0L)]
	public unsafe MjGlobalSettings()
		: this(IL2CPP.il2cpp_object_new(Il2CppClassPointerStore<MjGlobalSettings>.NativeClassPtr))
	{
		IntPtr* ptr = null;
		Unsafe.SkipInit(out IntPtr intPtr2);
		IntPtr intPtr = IL2CPP.il2cpp_runtime_invoke(NativeMethodInfoPtr__ctor_Public_Void_0, IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this), (void**)ptr, ref intPtr2);
		Il2CppException.RaiseExceptionIfNecessary(intPtr2);
	}

	public MjGlobalSettings(IntPtr pointer)
		: base(pointer)
	{
	}
}
