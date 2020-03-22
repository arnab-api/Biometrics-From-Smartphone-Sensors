package com.example.sensordemo;

import android.hardware.Sensor;

public class SensorInitialInfo {
    int sensor_type;
    String type_name;

    SensorInitialInfo(int sensor_type , String type_name){
        this.sensor_type = sensor_type;
        this.type_name = type_name;
    }
}
