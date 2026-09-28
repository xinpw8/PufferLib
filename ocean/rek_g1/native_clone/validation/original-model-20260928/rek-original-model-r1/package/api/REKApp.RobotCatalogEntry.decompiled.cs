using System;
using Il2CppInterop.Runtime;
using Il2CppInterop.Runtime.InteropTypes;
using Il2CppInterop.Runtime.Runtime;
using Il2CppSystem;
using UnityEngine;

namespace REKApp;

[System.Serializable]
public sealed class RobotCatalogEntry : Il2CppSystem.ValueType
{
	private static readonly System.IntPtr NativeFieldInfoPtr_id;

	private static readonly System.IntPtr NativeFieldInfoPtr_displayName;

	private static readonly System.IntPtr NativeFieldInfoPtr_typeLabel;

	private static readonly System.IntPtr NativeFieldInfoPtr_previewPrefab;

	private static readonly System.IntPtr NativeFieldInfoPtr_tileImage;

	private static readonly System.IntPtr NativeFieldInfoPtr_tileImageBW;

	private static readonly System.IntPtr NativeFieldInfoPtr_requiresOwnership;

	private static readonly System.IntPtr NativeFieldInfoPtr_requiresMatchingLoadout;

	private static readonly System.IntPtr NativeFieldInfoPtr_previewHeightOffset;

	public unsafe string id
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_id);
			return IL2CPP.Il2CppStringToManaged(*(System.IntPtr*)num);
		}
		set
		{
			System.IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (System.IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_id)), IL2CPP.ManagedStringToIl2Cpp(text));
		}
	}

	public unsafe string displayName
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_displayName);
			return IL2CPP.Il2CppStringToManaged(*(System.IntPtr*)num);
		}
		set
		{
			System.IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (System.IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_displayName)), IL2CPP.ManagedStringToIl2Cpp(text));
		}
	}

	public unsafe string typeLabel
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_typeLabel);
			return IL2CPP.Il2CppStringToManaged(*(System.IntPtr*)num);
		}
		set
		{
			System.IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (System.IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_typeLabel)), IL2CPP.ManagedStringToIl2Cpp(text));
		}
	}

	public unsafe GameObject previewPrefab
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_previewPrefab);
			System.IntPtr intPtr = *(System.IntPtr*)num;
			return (intPtr != (System.IntPtr)0) ? Il2CppObjectPool.Get<GameObject>(intPtr) : null;
		}
		set
		{
			System.IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (System.IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_previewPrefab)), IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)gameObject));
		}
	}

	public unsafe Texture2D tileImage
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_tileImage);
			System.IntPtr intPtr = *(System.IntPtr*)num;
			return (intPtr != (System.IntPtr)0) ? Il2CppObjectPool.Get<Texture2D>(intPtr) : null;
		}
		set
		{
			System.IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (System.IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_tileImage)), IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)texture2D));
		}
	}

	public unsafe Texture2D tileImageBW
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_tileImageBW);
			System.IntPtr intPtr = *(System.IntPtr*)num;
			return (intPtr != (System.IntPtr)0) ? Il2CppObjectPool.Get<Texture2D>(intPtr) : null;
		}
		set
		{
			System.IntPtr num = IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this);
			IL2CPP.il2cpp_gc_wbarrier_set_field(num, (System.IntPtr)((nint)num + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_tileImageBW)), IL2CPP.Il2CppObjectBaseToPtr((Il2CppObjectBase)(object)texture2D));
		}
	}

	public unsafe bool requiresOwnership
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_requiresOwnership);
			return *(bool*)num;
		}
		set
		{
			*(bool*)((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_requiresOwnership)) = flag;
		}
	}

	public unsafe bool requiresMatchingLoadout
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_requiresMatchingLoadout);
			return *(bool*)num;
		}
		set
		{
			*(bool*)((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_requiresMatchingLoadout)) = flag;
		}
	}

	public unsafe float previewHeightOffset
	{
		get
		{
			nint num = (nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_previewHeightOffset);
			return *(float*)num;
		}
		set
		{
			*(float*)((nint)IL2CPP.Il2CppObjectBaseToPtrNotNull((Il2CppObjectBase)(object)this) + (int)IL2CPP.il2cpp_field_get_offset(NativeFieldInfoPtr_previewHeightOffset)) = num;
		}
	}

	static RobotCatalogEntry()
	{
		Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr = IL2CPP.GetIl2CppClass("REKApp.dll", "REKApp", "RobotCatalogEntry");
		IL2CPP.il2cpp_runtime_class_init(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr);
		NativeFieldInfoPtr_id = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "id");
		NativeFieldInfoPtr_displayName = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "displayName");
		NativeFieldInfoPtr_typeLabel = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "typeLabel");
		NativeFieldInfoPtr_previewPrefab = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "previewPrefab");
		NativeFieldInfoPtr_tileImage = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "tileImage");
		NativeFieldInfoPtr_tileImageBW = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "tileImageBW");
		NativeFieldInfoPtr_requiresOwnership = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "requiresOwnership");
		NativeFieldInfoPtr_requiresMatchingLoadout = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "requiresMatchingLoadout");
		NativeFieldInfoPtr_previewHeightOffset = IL2CPP.GetIl2CppField(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr, "previewHeightOffset");
	}

	public RobotCatalogEntry(System.IntPtr pointer)
		: base(pointer)
	{
	}

	public RobotCatalogEntry()
		: base(IL2CPP.il2cpp_object_new(Il2CppClassPointerStore<RobotCatalogEntry>.NativeClassPtr))
	{
	}
}
