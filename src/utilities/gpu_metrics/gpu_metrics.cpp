#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

namespace py = pybind11;

std::string getZesErrorString(ze_result_t res) {
    return "Level Zero Error Code: " + std::to_string(res);
}

void checkZesResult(ze_result_t res, const char* operation) {
    if (res != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + getZesErrorString(res));
    }
}

// func to get all the gpu metrics
py::dict get_gpu_metrics() {
    py::dict results;

    // init sysman api
    ze_result_t res = zesInit(0);
    if (res != ZE_RESULT_SUCCESS) {
        return results;
    }

    uint32_t driverCount = 0;
    try {
        checkZesResult(zesDriverGet(&driverCount, nullptr), "zesDriverGet (count)");
    } catch (...) {
        return results;
    }

    if (driverCount == 0) return results;

    std::vector<zes_driver_handle_t> drivers(driverCount);
    checkZesResult(zesDriverGet(&driverCount, drivers.data()), "zesDriverGet");

    zes_driver_handle_t driver = drivers[0];

    uint32_t deviceCount = 0;
    checkZesResult(zesDeviceGet(driver, &deviceCount, nullptr), "zesDeviceGet (count)");
    if (deviceCount == 0) return results;

    std::vector<zes_device_handle_t> devices(deviceCount);
    checkZesResult(zesDeviceGet(driver, &deviceCount, devices.data()), "zesDeviceGet");

    for (uint32_t i = 0; i < deviceCount; ++i) {
        zes_device_handle_t sysmanDevice = devices[i];
        py::dict gpu_data;

        // device props
        zes_device_properties_t device_props = {};
        device_props.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (zesDeviceGetProperties(sysmanDevice, &device_props) == ZE_RESULT_SUCCESS) {
            gpu_data["name"] = std::string(device_props.core.name);
        } else {
            gpu_data["name"] = "Unknown Intel GPU";
        }

        // temperature (might be empty arr depending on your perms)
        uint32_t tempSensorCount = 0;
        if (zesDeviceEnumTemperatureSensors(sysmanDevice, &tempSensorCount, nullptr) == ZE_RESULT_SUCCESS && tempSensorCount > 0) {
            std::vector<zes_temp_handle_t> tempSensors(tempSensorCount);
            if (zesDeviceEnumTemperatureSensors(sysmanDevice, &tempSensorCount, tempSensors.data()) == ZE_RESULT_SUCCESS) {
                py::list temps;
                for (uint32_t j = 0; j < tempSensorCount; ++j) {
                    double temperature = 0.0;
                    if (zesTemperatureGetState(tempSensors[j], &temperature) == ZE_RESULT_SUCCESS) {
                        temps.append(temperature);
                    }
                }
                gpu_data["temperature"] = temps;
            }
        }

        // memory state
        uint32_t memCount = 0;
        if (zesDeviceEnumMemoryModules(sysmanDevice, &memCount, nullptr) == ZE_RESULT_SUCCESS && memCount > 0) {
            std::vector<zes_mem_handle_t> memories(memCount);
            if (zesDeviceEnumMemoryModules(sysmanDevice, &memCount, memories.data()) == ZE_RESULT_SUCCESS) {
                py::list mems;
                for (uint32_t j = 0; j < memCount; ++j) {
                    zes_mem_state_t mem_state = {};
                    mem_state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
                    if (zesMemoryGetState(memories[j], &mem_state) == ZE_RESULT_SUCCESS) {
                        py::dict mem_info;
                        mem_info["total"] = mem_state.size;
                        mem_info["free"] = mem_state.free;
                        mem_info["used"] = mem_state.size - mem_state.free;
                        mems.append(mem_info);
                    }
                }
                gpu_data["memory"] = mems;
            }
        }

        // power and energy counters
        uint32_t powerCount = 0;
        if (zesDeviceEnumPowerDomains(sysmanDevice, &powerCount, nullptr) == ZE_RESULT_SUCCESS && powerCount > 0) {
            std::vector<zes_pwr_handle_t> powers(powerCount);
            if (zesDeviceEnumPowerDomains(sysmanDevice, &powerCount, powers.data()) == ZE_RESULT_SUCCESS) {
                py::list pwrs;
                for (uint32_t j = 0; j < powerCount; ++j) {
                    zes_power_energy_counter_t pwr_state = {};
                    if (zesPowerGetEnergyCounter(powers[j], &pwr_state) == ZE_RESULT_SUCCESS) {
                        py::dict p_info;
                        p_info["energy"] = pwr_state.energy;
                        p_info["timestamp"] = pwr_state.timestamp;
                        pwrs.append(p_info);
                    }
                }
                gpu_data["power"] = pwrs;
            }
        }

        // engines (utilization / clocks)
        uint32_t engineCount = 0;
        if (zesDeviceEnumEngineGroups(sysmanDevice, &engineCount, nullptr) == ZE_RESULT_SUCCESS && engineCount > 0) {
            std::vector<zes_engine_handle_t> engines(engineCount);
            if (zesDeviceEnumEngineGroups(sysmanDevice, &engineCount, engines.data()) == ZE_RESULT_SUCCESS) {
                py::list engs;
                for (uint32_t j = 0; j < engineCount; ++j) {
                    zes_engine_stats_t eng_state = {};
                    if (zesEngineGetActivity(engines[j], &eng_state) == ZE_RESULT_SUCCESS) {
                        py::dict e_info;
                        e_info["active_time"] = eng_state.activeTime; // microsecs
                        e_info["timestamp"] = eng_state.timestamp; // microsecs

                        zes_engine_properties_t e_props = {};
                        e_props.stype = ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES;
                        if (zesEngineGetProperties(engines[j], &e_props) == ZE_RESULT_SUCCESS) {
                            e_info["type"] = static_cast<int>(e_props.type);
                        }

                        engs.append(e_info);
                    }
                }
                gpu_data["engines"] = engs;
            }
        }

        // frequency (clock speeds)
        uint32_t freqCount = 0;
        if (zesDeviceEnumFrequencyDomains(sysmanDevice, &freqCount, nullptr) == ZE_RESULT_SUCCESS && freqCount > 0) {
            std::vector<zes_freq_handle_t> freqs(freqCount);
            if (zesDeviceEnumFrequencyDomains(sysmanDevice, &freqCount, freqs.data()) == ZE_RESULT_SUCCESS) {
                py::list f_list;
                for(uint32_t j = 0; j < freqCount; ++j) {
                    zes_freq_state_t f_state = {};
                    f_state.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;
                    if(zesFrequencyGetState(freqs[j], &f_state) == ZE_RESULT_SUCCESS) {
                        f_list.append(f_state.actual); // mhz
                    }
                }
                gpu_data["clocks_mhz"] = f_list;
            }
        }

        results[std::to_string(i).c_str()] = gpu_data;
    }

    return results;
}

PYBIND11_MODULE(gpu_metrics, m) {
    m.doc() = "Intel GPU Metrics via Level Zero Sysman API";
    m.def("get_gpu_metrics", &get_gpu_metrics, "Fetch all available hardware metrics like temperature, memory, and clocks from Intel GPUs");
}
