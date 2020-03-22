package com.example.sensordemo;

import android.Manifest;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.File;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.util.EventListener;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {

    static final int REQUEST_PERMISSION_KEY = 1;
    SensorManager manager;
    Vector<Sensor> allSensors = new Vector<>();
    Vector<SensorEventListener>allListener = new Vector<>();
    Button record , reset , save , select_all;
    public static boolean is_recording = false;
    Vector<String> sensorData = new Vector<>();
    final String root_path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/SensorData/";
    TextView read_show;
    HashMap<CheckBox , SensorInitialInfo>check2sensor = new HashMap<>();
    Vector<SensorInitialInfo>activeSensors = new Vector<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        check2sensor.put((CheckBox)findViewById(R.id.Accelerometer), new SensorInitialInfo(Sensor.TYPE_ACCELEROMETER , "Accelerometer"));
        check2sensor.put((CheckBox)findViewById(R.id.Gravity), new SensorInitialInfo(Sensor.TYPE_GRAVITY , "Gravity"));
        check2sensor.put((CheckBox)findViewById(R.id.Gyroscope), new SensorInitialInfo(Sensor.TYPE_GYROSCOPE , "Gyroscope"));
        check2sensor.put((CheckBox)findViewById(R.id.Ambient_Temp), new SensorInitialInfo(Sensor.TYPE_AMBIENT_TEMPERATURE , "Ambient_Temp"));
        check2sensor.put((CheckBox)findViewById(R.id.Light), new SensorInitialInfo(Sensor.TYPE_LIGHT , "Light"));
        check2sensor.put((CheckBox)findViewById(R.id.Linear_Acceleration), new SensorInitialInfo(Sensor.TYPE_LINEAR_ACCELERATION , "Linear_Acceleration"));
        check2sensor.put((CheckBox)findViewById(R.id.Magnatic_Field), new SensorInitialInfo(Sensor.TYPE_MAGNETIC_FIELD , "Magnatic_Field"));
        check2sensor.put((CheckBox)findViewById(R.id.Orientation), new SensorInitialInfo(Sensor.TYPE_ORIENTATION , "Orientation"));
        check2sensor.put((CheckBox)findViewById(R.id.Pressure), new SensorInitialInfo(Sensor.TYPE_PRESSURE , "Pressure"));
        check2sensor.put((CheckBox)findViewById(R.id.Relative_Humidity), new SensorInitialInfo(Sensor.TYPE_RELATIVE_HUMIDITY , "Relative_Humidity"));
        check2sensor.put((CheckBox)findViewById(R.id.Rotation_Vector), new SensorInitialInfo(Sensor.TYPE_ROTATION_VECTOR , "Rotation_Vector"));
        check2sensor.put((CheckBox)findViewById(R.id.Temperature), new SensorInitialInfo(Sensor.TYPE_TEMPERATURE , "Temperature"));


        final String[] PERMISSIONS = {
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE,
        };

        Log.d(ProjectConstants.TAG, "permissions:: ");
        for (String str : PERMISSIONS) Log.d("___sdk", " ------> " + str);

        if (!Utils.hasPermissions(this, PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_PERMISSION_KEY);
        }


        manager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        List<Sensor> sensorList = manager.getSensorList(Sensor.TYPE_ALL);

        for(Sensor sensor : sensorList){
            Log.d(ProjectConstants.TAG , "FOUND SENSOR -----> :: " + sensor.getName());
        }

        is_recording = false;
        record = (Button)findViewById(R.id.record);
        record.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(is_recording == false){
                    Log.d(ProjectConstants.TAG , " ***** RECORDING *****");
                    showToastNotification(" ***** RECORDING *****");
                    is_recording = true;
                    record.setText("PAUSE");
                    activeSensors.clear();

                    for(CheckBox sensor_check : check2sensor.keySet()){
                        if(sensor_check.isChecked()) activeSensors.add(check2sensor.get(sensor_check));
                        sensor_check.setClickable(false);
                    }
                    select_all.setVisibility(View.INVISIBLE); select_all.setClickable(false);
                    reset.setVisibility(View.INVISIBLE); reset.setClickable(false);
                    save.setVisibility(View.INVISIBLE); save.setClickable(false);

                    registerSensorListeners();
                }
                else{
                    Log.d(ProjectConstants.TAG , " ***** STOPPING *****");
                    showToastNotification(" ***** STOPPING *****");
                    is_recording = false;
                    record.setText("RECORD");
                    unregisterSensorListeners();

                    for(CheckBox sensor_check : check2sensor.keySet()){
                        sensor_check.setClickable(true);
                    }
                    select_all.setVisibility(View.VISIBLE); select_all.setClickable(true);
                    reset.setVisibility(View.VISIBLE); reset.setClickable(true);
                    save.setVisibility(View.VISIBLE); save.setClickable(true);
                }
            }
        });

        reset = (Button)findViewById(R.id.reset);
        reset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sensorData.clear();
                read_show.setText(String.format("%09d" , sensorData.size()));
                for(CheckBox sensor_check : check2sensor.keySet()){
                    sensor_check.setChecked(false);
                }
                showToastNotification("data reset done successfully");
            }
        });

        save = (Button)findViewById(R.id.save);
        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {

                    makeRootFolder();
                    Timestamp ts = new Timestamp(System.currentTimeMillis());
                    String file_name = "sample_" + ts + ".json";

                    PrintWriter writer = new PrintWriter(root_path + file_name);

                    writer.print("[");
                    boolean first = true;
                    for(String reading : sensorData){
                        if(first == true) first = false;
                        else writer.print(",");
                        writer.print(reading);
                    }
                    writer.print("]");
                    writer.close();

                    showToastNotification("data saved @ " + file_name);
                }catch(Exception e){
                    showToastNotification("ERROR!! Problem during saving data");
                }
            }
        });

        read_show = (TextView)findViewById(R.id.read_cnt);

        select_all = (Button)findViewById(R.id.select_all);
        select_all.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
               for(CheckBox sensor_check : check2sensor.keySet()){
                   sensor_check.setChecked(true);
               }
            }
        });
    }

    void makeRootFolder() {
        String path = root_path;
        File root = new File(path);
        if (!root.exists()) {
            if (!root.mkdirs()) {
                Log.d(ProjectConstants.TAG, "failed to create directory");
            } else Log.d(ProjectConstants.TAG, "Created Directory");
        } else Log.d(ProjectConstants.TAG, " directory already exists");
    }

    JsonObject convertToJSON(Object data){
        return convertToJSON(data , null);
    }
    JsonObject convertToJSON(Object data, final String sensorName) {
        Gson gson = new Gson();
        String json = gson.toJson(data);
        //json = json.replace("[" , "{").replace("]" , "}");


        JsonObject obj = new JsonObject();
        obj.addProperty("vector", json);

        if(sensorName != null) obj.addProperty("sensor", sensorName);
        return obj;
    }

    SensorEventListener getEventListener() {
        return getEventListener("N/A");
    }
    SensorEventListener getEventListener(final String sensorName) {
        SensorEventListener eventListener = new SensorEventListener() {
            @Override
            public void onSensorChanged(SensorEvent event) {
                String output = sensorName + " values :: ";
                for(float val : event.values){
                    output += String.format(" %.3f" , val);
                }

                JsonObject obj = convertToJSON(event.values , sensorName);
                Gson gson = new Gson();
                String json = gson.toJson(obj);
                sensorData.add(json);

                read_show.setText(String.format("%09d" , sensorData.size()));

                Log.d(ProjectConstants.TAG , sensorName + ": " + obj);
            }

            @Override
            public void onAccuracyChanged(Sensor sensor, int accuracy) {

            }
        };
        return eventListener;
    }

    void registerSensorListeners(){
        for(SensorInitialInfo info : activeSensors){
            Sensor sensor = manager.getDefaultSensor(info.sensor_type);
            if(sensor != null){
                try {
                    allSensors.add(sensor);
                    SensorEventListener listener = getEventListener(info.type_name);
                    allListener.add(listener);
                    manager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_NORMAL);
                    Log.d(ProjectConstants.TAG, "SENSOR ACTIVATED >>> " + info.type_name);
                }catch(Exception e){
                    Log.d(ProjectConstants.TAG , "ERROR registering listener to " + info.type_name);
                }
            }else{
                Log.d(ProjectConstants.TAG , "COULD NOT FIND SENSOR >>> " + info.type_name);
            }
        }
    }

    void unregisterSensorListeners(){
        for(SensorEventListener listener: allListener){
            if(listener != null){
                try{
                    manager.unregisterListener(listener);
                }catch(Exception e){
                    Log.d(ProjectConstants.TAG , "Problem unregistering listener :: ");
                }
            }
        }
    }

    void showToastNotification(String msg) {
        Toast.makeText(getApplicationContext(), "Notification: " + msg + " !!",
                Toast.LENGTH_SHORT).show();
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterSensorListeners();
    }
}
