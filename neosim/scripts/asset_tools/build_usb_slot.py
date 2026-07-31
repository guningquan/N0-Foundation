"""Build the USB mating socket (base with hole + lead-in chamfer). Hole = connector cross-section + per-side clearance.
Usage: python scripts/asset_tools/build_usb_slot.py [out.obj]   (default /tmp/usb_assets/USBSlot.obj)
"""
import sys, os, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usb_assets/USBSlot.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

CW, CT = 12.0, 4.5            # connector cross-section (mm)
CLR = 1.5                     # per-side clearance: must exceed IPC d_hat (0.5mm) or insertion jams
hw, ht = CW + 2 * CLR, CT + 2 * CLR
H, D = 13.0, 11.0            # base height / hole depth
LEAD = 0.8                    # lead-in chamfer: top widened by 0.8mm per side, 1.5mm deep

outer = trimesh.creation.box([26, 16, H]);              outer.apply_translation([0, 0, H / 2])
main = trimesh.creation.box([hw, ht, 2 * D]);           main.apply_translation([0, 0, H])
lead = trimesh.creation.box([hw + 2 * LEAD, ht + 2 * LEAD, 3.0]); lead.apply_translation([0, 0, H])
slot = trimesh.boolean.difference([outer, main, lead])
slot.apply_scale(0.001)
slot.export(out)
print(f"saved {out}  hole(mm)={hw}x{ht} depth={D}  watertight={slot.is_watertight}")
