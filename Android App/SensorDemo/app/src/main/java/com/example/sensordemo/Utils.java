package com.example.sensordemo;

import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.ActivityCompat;

import java.util.Calendar;
import java.util.Date;

public class Utils {
    public static  boolean hasPermissions(Context context, String... permissions) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    public static String getFormattedTime() {
        Date c = Calendar.getInstance().getTime();
        String formattedTime = String.format("%02d:%02d:%02d",c.getHours(), c.getMinutes(), c.getSeconds());

        return formattedTime;
    }
}