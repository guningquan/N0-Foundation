#!/usr/bin/env bash
# Build and register VK_LAYER_N0VTLA_clamp_maxalloc, a user-level implicit Vulkan layer (no root).
#
# NVIDIA driver 595.x reports maxMemoryAllocationSize = UINT64_MAX. Isaac Sim 4.5's RTX scene DB
# truncates it to a signed 32-bit size and segfaults at startup (librtx.scenedb.plugin.so,
# "SceneDbContext : Buffer size instance limit -256"). The layer clamps that one value and passes
# everything else through. It is opt-in: only active when VK_CLAMP_MAXALLOC=1 is set, so other
# Vulkan programs are unaffected. See docs/ServerSetupNotes.md.
#
#   usage: bash tools/vk_clamp_maxalloc/install.sh [install_dir (default: ~/.local/lib/vk_clamp_maxalloc)]
#   then:  export VK_CLAMP_MAXALLOC=1        # optional: VK_CLAMP_MAXALLOC_VALUE (default 0xFFFFFFFF)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:-$HOME/.local/lib/vk_clamp_maxalloc}
LAYER_DIR=$HOME/.local/share/vulkan/implicit_layer.d
LIB=$DEST/libVkLayer_n0vtla_clamp_maxalloc.so
mkdir -p "$DEST" "$LAYER_DIR"

# System gcc + the Vulkan headers in /usr/include (an active conda env may shadow gcc)
/usr/bin/gcc -shared -fPIC -O2 -Wall -o "$LIB" "$HERE/clamp_maxalloc.c" -lpthread

cat > "$LAYER_DIR/VkLayer_n0vtla_clamp_maxalloc.json" <<EOF
{
    "file_format_version": "1.1.2",
    "layer": {
        "name": "VK_LAYER_N0VTLA_clamp_maxalloc",
        "type": "GLOBAL",
        "library_path": "$LIB",
        "api_version": "1.3.204",
        "implementation_version": "1",
        "description": "Clamp maxMemoryAllocationSize (driver 595 reports UINT64_MAX; Isaac Sim 4.5 RTX scene DB crashes on it)",
        "enable_environment": { "VK_CLAMP_MAXALLOC": "1" },
        "disable_environment": { "VK_CLAMP_MAXALLOC_DISABLE": "1" }
    }
}
EOF
echo "installed $LIB"
echo "registered $LAYER_DIR/VkLayer_n0vtla_clamp_maxalloc.json"
echo "check: VK_CLAMP_MAXALLOC=1 vulkaninfo 2>&1 | grep -m1 clamp_maxalloc"
