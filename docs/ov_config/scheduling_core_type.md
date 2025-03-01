

`ov::hint::scheduling_core_type` specifies the type of CPU cores for CPU inference when the user runs inference on a hybird platform that includes both Performance-cores (P-cores) and Efficient-cores (E-cores). If the user platform only has one type of CPU core, this property has no effect, and CPU inference always uses this unique core type.