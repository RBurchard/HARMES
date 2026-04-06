
config = {
    "window_sizes": [5, 10],
    "sensor_configs": [["all"], ["imu_l"], ["imu_r"], ["imu_l", "imu_r"], ["a"], ["imu_r", "a"], ["imu_l", "imu_r", "a"],
                       ["a", "h"], ["imu_l", "imu_r", "h"]],
    ### training:
    "batch_size": 32,
    "epochs": 35
}

base_path = "/home/rb995633/AudioDS/Dataset/Records"
