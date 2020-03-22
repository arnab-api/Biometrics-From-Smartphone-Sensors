
window.onload = function () {
    // TODO:: Do your initialization job

    function requestAPermission(req){
        
        function onsuccessPermission(result , privilege){
            console.log("permission: " + req + " :: " , result);
        }
        function onErrorPermission(error){
            console.log("permission: " + req + " error: " , error);
        }
        tizen.ppm.requestPermission(req , onsuccessPermission , onErrorPermission);
    }

    function requestAllPermissions(){
        requestAPermission("http://developer.samsung.com/privilege/healthinfo");
        requestAPermission("http://tizen.org/privilege/healthinfo");
        requestAPermission("http://tizen.org/privilege/filesystem.read");
        requestAPermission("http://tizen.org/privilege/filesystem.write");
        requestAPermission("http://tizen.org/privilege/mediastorage");
    }

    allSensors = {
    		
    };
    
    sensorReadings = [];
    startTime = Date.now();

    // sensor data has non enumerable properties
    // JSON.stringify() can not see non enum properties
    // So sensor data must be converted to base object
    //  ---- arnab.sharma
    function convert2Object(inp){
        var properties = Object.getOwnPropertyNames(inp);
        var obj = {};

        for(var i = 0 ; i < properties.length ; i++){
            var p = properties[i];
            obj[p] = inp[p];
        }
        return obj;
    }

    function registerSensor(sensorName){
        try{
            var sensor = tizen.sensorservice.getDefaultSensor(sensorName);
            function onsuccessCB() {
                console.log(sensorName + ' service has started successfully.');
            }

            function onchangedCB(sensorData) {

                var obj = convert2Object(sensorData);

                reading = {
                    'Date&time': new Date().toLocaleString(),
                    'time': Date.now() - startTime,
                    'sensorType': sensorName,
                    'reading': obj
                };
                // console.log(sensorName + ' sensor data: ' , sensorData);
                console.log(reading , JSON.stringify(reading));
                sensorReadings.push(reading);
            }

            sensor.setChangeListener(onchangedCB);
            sensor.start(onsuccessCB);
            
            allSensors[sensorName] = sensor;
            // console.log(sensor);
        }catch(ex){
            console.log(sensorName + " failed to start");
            console.log(ex);
        }   
    }
    
    function startHumanActivityHRM(sensorType){

        try{
            function onchangedCB(hrmInfo) {
                var obj = convert2Object(hrmInfo);

                reading = {
                    'Date&time': new Date().toLocaleString(),
                    'time': Date.now() - startTime,
                    'sensorType': sensorType,
                    'reading': obj
                };
                // console.log(sensorName + ' sensor data: ' , sensorData);
                // console.log(hrmInfo , JSON.stringify(hrmInfo));
                // console.log(obj , JSON.stringify(obj));
                console.log(sensorType + " :: " , reading , JSON.stringify(reading));
                sensorReadings.push(reading);
            }

            tizen.humanactivitymonitor.start(sensorType, onchangedCB);
            console.log(sensorType , " started successfully");
        }catch(ex){
            console.log(sensorType , "failed to start::" , ex);
        }
    }

    function startPedometer(){
        function onchangedCB(pedometerInfo) {
            var obj = convert2Object(pedometerInfo);

            reading = {
                'Date&time': new Date().toLocaleString(),
                'time': Date.now() - startTime,
                'sensorType': 'PEDOMETER',
                'reading': obj
            };

            // console.log(typeof(reading) , reading , JSON.stringify(reading));
            sensorReadings.push(reading);            
            
        }
        
        /* Deregisters a previously registered listener */
        tizen.humanactivitymonitor.unsetAccumulativePedometerListener();
        
        tizen.humanactivitymonitor.setAccumulativePedometerListener(onchangedCB);
    }
    
    function recognizeHumanActivity(activity){
        function errorCallback(error) {
            console.log(error.name + ': ' + error.message);
        }
        
        function listener(info) {
            // console.log(activity + " :: " , info);
            var obj = convert2Object(info);
            reading = {
                'Date&time': new Date().toLocaleString(),
                'time': Date.now() - startTime,
                'sensorType': "HUMAN_ACT_" + activity,
                'reading': obj
            };

            // console.log(typeof(reading) , reading , JSON.stringify(reading));
            sensorReadings.push(reading); 
        }
        
        try {
            var listenerId = tizen.humanactivitymonitor.addActivityRecognitionListener(activity, listener, errorCallback);
            console.log(activity , " started successfully");
        } catch (error) {
            console.log(activity , " failed to start :: " , error);
        }
    }

    function initializeSensors(){
        sensors = tizen.sensorservice.getAvailableSensors();
        console.log("All Sensors: " + sensors);
    
        for (var s = 0 ; s < sensors.length ; s++){
            registerSensor(sensors[s]);
        }
        startHumanActivityHRM("HRM");
        startHumanActivityHRM("SLEEP_MONITOR");
        startHumanActivityHRM("WRIST_UP");
        startPedometer();
        
        var activities = [
            'STATIONARY',
            'WALKING',
            'RUNNING',
            'IN_VEHICLE'
        ];
        for(var i = 0 ; i < activities.length ; i++) {
            recognizeHumanActivity(activities[i]);
        }

        console.log(allSensors);
    }
    
    function reset(){
        sensorReadings = [];
    }

    function stopRunningSensors(){
        for (var s = 0 ; s < sensors.length ; s++){
            var sensorName = sensors[s]
            if(allSensors[sensorName] != null) {
                allSensors[sensorName].stop();
            }
        }
        try{ 
            tizen.humanactivitymonitor.stop('HRM');
        }catch(ex){
            console.log("STOP: Exception:" , ex);
        }
        try{ 
            tizen.humanactivitymonitor.unsetAccumulativePedometerListener();
        }catch(ex){
            console.log("STOP: Exception:" , ex);
        }
    }
    
    function saveCurrentData(){
        console.log("saving data");
        console.log(">>>" , fileManager);
        
        filename = Date.now() + ".json";
        fileManager.createFile(filename);

        var file_obj;
        file_obj = fileManager.resolve(filename);
        
        file_obj.openStream(
            'w',
            function(fileStream) {
                var content = JSON.stringify(sensorReadings);
                fileStream.write(content);
                fileStream.close();
            },
            function(error) {
                console.log(JSON.stringify(error));
            }
        );
        console.log("SAVED SUCCESSFULLY");
    }

    // ------ MAIN START ------
    requestAllPermissions();
    
    fileManagerInit = false;
    function initFileManager(){
        tizen.filesystem.resolve(
            'documents',
            function(obj) {
                fileManager = obj;
                console.log("-.-.-." , fileManager);
            },
            function(error) {
                console.log('Error::', JSON.stringify(error));
            },"rw"
        );
        console.log("function called");
    }


    // add eventListener for tizenhwkey
    document.addEventListener('tizenhwkey', function(e) {
        if(e.keyName == "back")
        try {
            tizen.application.getCurrentApplication().exit();
        } catch (ignore) {
	    }
    });
    
    
    fileManager = null;

    var textbox = document.querySelector('.contents');
    textbox.addEventListener("click", function(){
    	box = document.querySelector('#textbox');
    	console.log("click event" , box.innerHTML)
    	if(box.innerHTML == "Start"){
            initializeSensors();
            
            if(fileManagerInit == false){
                initFileManager();
                setTimeout(function(){
                    console.log("after function call::" , fileManager);
                } , 1000);
                fileManagerInit = true;
            }

            box.innerHTML = "Stop";
        }
        else{
            console.log("stopping running sensors");
            stopRunningSensors();
            box.innerHTML = "Start";
        }
    });

    var savebox = document.querySelector('#save');
    savebox.addEventListener("click", function(){
        box3 = document.querySelector('#save');
        var msg = "success";
        if(box3.innerHTML == "save"){
            console.log("save called");
            try{
                saveCurrentData();
            }catch(ex){
                msg = "error";
                console.log("error saving data:: " , ex);
            }
            box3.innerHTML = msg;
        }
        else{
            box3.innerHTML = "save";
        }        
    });

    var resetbox = document.querySelector('#reset');
    resetbox.addEventListener("click", function(){
        box2 = document.querySelector('#reset');
        if(box2.innerHTML == "reset"){
            console.log("reset called");
            reset();
            box2.innerHTML = "success";
        }
        else{
            box2.innerHTML = "reset";
        }        
    });
};

// sensors = [
//     ACCELERATION,
//     GRAVITY,
//     LINEAR_ACCELERATION,
//     GYROSCOPE,
//     LIGHT,
//     PRESSURE,
//     GYROSCOPE_UNCALIBRATED,
//     GYROSCOPE_ROTATION_VECTOR,
//     HRM_RAW
// ]
