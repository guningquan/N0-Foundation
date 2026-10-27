// VK_LAYER_N0VTLA_clamp_maxalloc
//
// NVIDIA driver 595.84 reports VkPhysicalDeviceMaintenance3Properties::maxMemoryAllocationSize as
// UINT64_MAX. Isaac Sim 4.5's RTX scene DB (librtx.scenedb.plugin.so) truncates it to a signed 32-bit
// size ("Buffer size instance limit -256"; 2^40 gives 0) and segfaults while creating the hydra
// engine. This layer clamps that one value (in both Maintenance3 and Vulkan11 properties) to
// VK_CLAMP_MAXALLOC_VALUE (default 0xFFFFFFFF; anything below 2^39 works) and passes everything else
// through unchanged.
//
// Implicit layer, opt-in: the manifest only enables it when VK_CLAMP_MAXALLOC=1.
// Build: gcc -shared -fPIC -O2 -o libVkLayer_n0vtla_clamp_maxalloc.so clamp_maxalloc.c -lpthread

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#define LAYER_EXPORT __attribute__((visibility("default")))
#define DEFAULT_CLAMP_VALUE 0xFFFFFFFFULL
#define MAX_OBJECTS 64

typedef struct {
    void *key;
    VkInstance instance;
    PFN_vkGetInstanceProcAddr next_gipa;
    PFN_vkDestroyInstance next_destroy;
    PFN_vkGetPhysicalDeviceProperties2 next_props2;
    PFN_vkGetPhysicalDeviceProperties2 next_props2_khr;
} InstanceData;

typedef struct {
    void *key;
    PFN_vkGetDeviceProcAddr next_gdpa;
    PFN_vkDestroyDevice next_destroy;
} DeviceData;

static InstanceData g_instances[MAX_OBJECTS];
static DeviceData g_devices[MAX_OBJECTS];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_reported = 0;

static VkDeviceSize clamp_value(void) {
    static VkDeviceSize value = 0;
    if (value == 0) {
        const char *env = getenv("VK_CLAMP_MAXALLOC_VALUE");
        value = env ? (VkDeviceSize)strtoull(env, NULL, 0) : 0;
        if (value == 0) value = DEFAULT_CLAMP_VALUE;
    }
    return value;
}

// Dispatchable handles start with a pointer to the loader dispatch table; physical devices share
// their instance's table, so the same key finds the instance data.
static void *dispatch_key(const void *handle) { return *(void *const *)handle; }

static InstanceData *find_instance(void *key) {
    for (int i = 0; i < MAX_OBJECTS; ++i)
        if (g_instances[i].key == key) return &g_instances[i];
    return NULL;
}

static DeviceData *find_device(void *key) {
    for (int i = 0; i < MAX_OBJECTS; ++i)
        if (g_devices[i].key == key) return &g_devices[i];
    return NULL;
}

static void clamp_chain(VkPhysicalDeviceProperties2 *props) {
    for (VkBaseOutStructure *s = (VkBaseOutStructure *)props->pNext; s != NULL; s = s->pNext) {
        VkDeviceSize *size = NULL;
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
            size = &((VkPhysicalDeviceMaintenance3Properties *)s)->maxMemoryAllocationSize;
        else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES)
            size = &((VkPhysicalDeviceVulkan11Properties *)s)->maxMemoryAllocationSize;
        if (size != NULL && *size > clamp_value()) {
            if (!g_reported) {
                fprintf(stderr, "[clamp_maxalloc] maxMemoryAllocationSize 0x%llx -> 0x%llx\n",
                        (unsigned long long)*size, (unsigned long long)clamp_value());
                g_reported = 1;
            }
            *size = clamp_value();
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL layer_GetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2 *props) {
    pthread_mutex_lock(&g_lock);
    InstanceData *d = find_instance(dispatch_key(pd));
    PFN_vkGetPhysicalDeviceProperties2 next = d ? d->next_props2 : NULL;
    pthread_mutex_unlock(&g_lock);
    if (next == NULL) return;
    next(pd, props);
    clamp_chain(props);
}

static VKAPI_ATTR void VKAPI_CALL layer_GetPhysicalDeviceProperties2KHR(VkPhysicalDevice pd, VkPhysicalDeviceProperties2 *props) {
    pthread_mutex_lock(&g_lock);
    InstanceData *d = find_instance(dispatch_key(pd));
    PFN_vkGetPhysicalDeviceProperties2 next = d ? (d->next_props2_khr ? d->next_props2_khr : d->next_props2) : NULL;
    pthread_mutex_unlock(&g_lock);
    if (next == NULL) return;
    next(pd, props);
    clamp_chain(props);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *alloc, VkInstance *instance) {
    VkLayerInstanceCreateInfo *link = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerInstanceCreateInfo *)link->pNext;
    if (link == NULL) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;  // advance the chain for the next layer
    PFN_vkCreateInstance next_create = (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult result = next_create(ci, alloc, instance);
    if (result != VK_SUCCESS) return result;

    pthread_mutex_lock(&g_lock);
    InstanceData *d = find_instance(NULL);
    if (d != NULL) {
        d->key = dispatch_key(*instance);
        d->instance = *instance;
        d->next_gipa = next_gipa;
        d->next_destroy = (PFN_vkDestroyInstance)next_gipa(*instance, "vkDestroyInstance");
        d->next_props2 = (PFN_vkGetPhysicalDeviceProperties2)next_gipa(*instance, "vkGetPhysicalDeviceProperties2");
        d->next_props2_khr = (PFN_vkGetPhysicalDeviceProperties2)next_gipa(*instance, "vkGetPhysicalDeviceProperties2KHR");
    }
    pthread_mutex_unlock(&g_lock);
    return d != NULL ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&g_lock);
    InstanceData *d = find_instance(dispatch_key(instance));
    PFN_vkDestroyInstance next = d ? d->next_destroy : NULL;
    if (d != NULL) memset(d, 0, sizeof(*d));
    pthread_mutex_unlock(&g_lock);
    if (next != NULL) next(instance, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *alloc, VkDevice *device) {
    VkLayerDeviceCreateInfo *link = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerDeviceCreateInfo *)link->pNext;
    if (link == NULL) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;  // advance the chain for the next layer

    pthread_mutex_lock(&g_lock);
    InstanceData *inst = find_instance(dispatch_key(pd));
    VkInstance instance = inst ? inst->instance : VK_NULL_HANDLE;
    pthread_mutex_unlock(&g_lock);
    PFN_vkCreateDevice next_create = (PFN_vkCreateDevice)next_gipa(instance, "vkCreateDevice");
    if (next_create == NULL) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult result = next_create(pd, ci, alloc, device);
    if (result != VK_SUCCESS) return result;

    pthread_mutex_lock(&g_lock);
    DeviceData *d = find_device(NULL);
    if (d != NULL) {
        d->key = dispatch_key(*device);
        d->next_gdpa = next_gdpa;
        d->next_destroy = (PFN_vkDestroyDevice)next_gdpa(*device, "vkDestroyDevice");
    }
    pthread_mutex_unlock(&g_lock);
    return d != NULL ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR void VKAPI_CALL layer_DestroyDevice(VkDevice device, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&g_lock);
    DeviceData *d = find_device(dispatch_key(device));
    PFN_vkDestroyDevice next = d ? d->next_destroy : NULL;
    if (d != NULL) memset(d, 0, sizeof(*d));
    pthread_mutex_unlock(&g_lock);
    if (next != NULL) next(device, alloc);
}

LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name);

LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)layer_CreateInstance;
    if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)layer_DestroyInstance;
    if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)layer_CreateDevice;
    if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)layer_DestroyDevice;
    if (!strcmp(name, "vkGetPhysicalDeviceProperties2")) return (PFN_vkVoidFunction)layer_GetPhysicalDeviceProperties2;
    if (!strcmp(name, "vkGetPhysicalDeviceProperties2KHR")) return (PFN_vkVoidFunction)layer_GetPhysicalDeviceProperties2KHR;
    if (instance == VK_NULL_HANDLE) return NULL;
    pthread_mutex_lock(&g_lock);
    InstanceData *d = find_instance(dispatch_key(instance));
    PFN_vkGetInstanceProcAddr next = d ? d->next_gipa : NULL;
    pthread_mutex_unlock(&g_lock);
    return next ? next(instance, name) : NULL;
}

LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)layer_DestroyDevice;
    pthread_mutex_lock(&g_lock);
    DeviceData *d = find_device(dispatch_key(device));
    PFN_vkGetDeviceProcAddr next = d ? d->next_gdpa : NULL;
    pthread_mutex_unlock(&g_lock);
    return next ? next(device, name) : NULL;
}

LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *iface) {
    if (iface == NULL || iface->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (iface->loaderLayerInterfaceVersion >= 2) {
        iface->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        iface->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        iface->pfnGetPhysicalDeviceProcAddr = NULL;
        iface->loaderLayerInterfaceVersion = 2;
    }
    return VK_SUCCESS;
}
