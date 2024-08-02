# 
#
# python /tmp/GT_bmc_pyutil.py -c Get_Telemetries
# python /tmp/GT_bmc_pyutil.py -c Get_TelemetriesRetimers
# python /tmp/GT_bmc_pyutil.py -c RF_FetchDebugLogs
#
#
#
import sys, json, time, datetime
import optparse
import urllib.request, urllib.parse
from urllib import request, parse
import traceback
# import requests


def main():

    parser = optparse.OptionParser()


    parser.add_option('-q', '--query',
        action="store", dest="query",
        help="query string", default="spam")

    parser.add_option('-t', '--test',
        action="store", dest="test",
        help="test string", default="test123")

    parser.add_option('-c', '--command',
        action="store", dest="cmd",
        help="command string", default="help")

    parser.add_option('-a', '--argument',
        action="store", dest="arg",
        help="arg string", default="")

    options, args = parser.parse_args()

    # print ('Query string:', options.query)
    # print ('test string:', options.test)
    print ('command string:', options.cmd)
    if (options.cmd=="Get_Retimer0Temp"):  
        Get_Retimer0Temp();
    elif (options.cmd=="Get_AllRetimerTemp"):  
        Get_AllRetimerTemp();
    elif (options.cmd=="Get_Telemetries"):  
        Get_Telemetries();
    elif (options.cmd=="Get_TelemetriesRetimers"):  
        Get_TelemetriesRetimers();
    elif (options.cmd=="Get_EventLogs"):  
        Get_EventLogs();
    elif (options.cmd=="Get_EventLogs_Sorted"):  
        Get_EventLogs_Sorted();
    elif (options.cmd=="RF_FetchDebugLogs"):  
        if options.arg=="":
            RF_FetchDebugLogs();
        else:
            RF_FetchDebugLogs(sDataType=options.arg);
    else:
        print("Error unhandle command", options.cmd);

def RF_FetchDebugLogs(sDataType = "AllLogs"):
    """
        Desriptions:
            Fetch debug logs from SMC with RF commands
        Test Command in BMC:
            python3 /tmp/GT_bmc_pyutil.py -c RF_FetchDebugLogs
            python3 /tmp/GT_bmc_pyutil.py -c RF_FetchDebugLogs -a AllCPERs
            python3 /tmp/GT_bmc_pyutil.py -c RF_FetchDebugLogs -a JournalControl
    

            From the schema spec: 
                
                "OEMDiagnosticDataType@Redfish.AllowableValues": [
                    "Dmesg",
                    "JournalControl",
                    "InternalLogServices",
                    "NetworkStatus",
                    "SysLog",
                    "TempSensorsRegisters",
                    "FPGADump",
                    "GPURegisters",
                    "RMRegisters",
                    "RetimerDump",
                    "OAMandUBBRegCfg",
                    "OAMandUBBThrmMeas",
                    "UBBandOAMFRU",
                    "FWVersions",
                    "AllLogs",
                    "AllCPERs",
                    "EROTLogs",
                    "RetLTSSM",
                    "APMLAllOAM"
    """
    try:
        sURL = "http://192.168.31.1/redfish/v1/Systems/UBB/LogServices/DiagLogs/Actions/LogService.CollectDiagnosticData"
        data = {"DiagnosticDataType":"OEM", "OEMDiagnosticDataType" : sDataType  }
        # enc_data = parse.urlencode(data).encode()
        enc_data = str(json.dumps(data)).encode('utf-8');
        nDelay = 5
        if (sDataType=="AllLogs"):
            nDelay = 30
            
        print("sURL = %s" %(sURL)) 
        print("enc_data = %s" %(enc_data)) 
        req = request.Request(sURL, data=enc_data)
        req.add_header('Content-Type', 'application/json')
        # req = request.Request(sURL, data=enc_data)
        with request.urlopen(req) as resp:
            # print ("resp.status_code : ",  resp.status_code)
            j = json.load(resp)
            # print ("resp.json : ",  resp.json())
            print(json.dumps(j, indent=4))
            sURL = "http://192.168.31.1" + j['@odata.id']
            print("sURL = %s" %(sURL)) 
            while j['TaskStatus']=='OK' and j['TaskState']=='Running':
                with urllib.request.urlopen(sURL) as u2:
                    j = json.load(u2)
                    # print(json.dumps(j, indent=4))
                    sNow = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S' );
                    print("sURL=%s - %s --- TaskState=%s - TaskStatus=%s" %(sURL, sNow, j["TaskState"], j["TaskStatus"]))
                if j['TaskState']!='Completed':
                    time.sleep(nDelay);

            # print(json.dumps(j, indent=4))
            if j['TaskState']=='Completed':
                if j['TaskStatus']!='OK':
                    print("TaskStatus!=OK")
                    print("TaskStatus!=OK")
                    print("TaskStatus!=OK")
                    print(json.dumps(j, indent=4))
                    time.sleep(nDelay);
                PayLoad_HttpHeaders = j['Payload']['HttpHeaders']
                # Now scan the Payloads HttpHeaders for Location...  Argh...
                for s in PayLoad_HttpHeaders:
                    # print(s)
                    if (s.startswith("Location: ")):
                        sLoc = s[len("Location: "):]
                        i = 0; 
                        while True:
                            sFileName = datetime.datetime.now().strftime(sDataType + '---%Y-%m-%d--%H-%M-%S.tar.xz'  );
                            print("sLoc=%s   sFileName=%s" %(sLoc, sFileName))
                            try:
                                urllib.request.urlretrieve("http://192.168.31.1" + sLoc+'/attachment', sFileName);
                                break;
                            except:
                                i = i + 1 
                                traceback.print_exc();
                                time.sleep(60);
                                if (i>10):  
                                    print("Unable to fetch %s after 10 mins, give up." %("http://192.168.31.1" + sLoc+'/attachment')) 
                                    print("Unable to fetch %s after 10 mins, give up." %("http://192.168.31.1" + sLoc+'/attachment')) 
                                    print("Unable to fetch %s after 10 mins, give up." %("http://192.168.31.1" + sLoc+'/attachment')) 
                                    break;
                                pass

            else:  # if j['TaskState']=='Completed':
                print('RF_FetchDebugLogs: Task not Completed Something is wrong... ')
                print('RF_FetchDebugLogs: Task not Completed Something is wrong... ')
                print('RF_FetchDebugLogs: Task not Completed Something is wrong... ')
                print(json.dumps(j, indent=4))
        return 0;

    except:
        traceback.print_exc();
        return -1;


def Get_AllRetimerTemp():
    """
        Read Retimer Temp and provide output in tabulated format
        bmc shell monitor command for long period of time:  
            for I in {0..2000} ; do py_Get_AllRetimerTemp;  sleep 20 ; done
    """
    try:
        print("Retimer Temp: ", end='' )
        for i in range(0,8):
            s = "http://192.168.31.1/redfish/v1/Chassis/RETIMER_%d/Sensors/RETIMER_%d_TEMP" % (i, i);
            with urllib.request.urlopen(s) as url:
                j= json.load(url)
                # print(j)
                if (j['Status']['Health']!='OK' or j['Status']['State']!='Enabled'): 
                    print("Retimer: Health not OK or State no enabled");
                    print("Retimer: Health not OK or State no enabled");
                    print("Retimer: Health not OK or State no enabled");
                    print(json.dumps(j, indent=4))
                f = float(j['Reading']);
                print("  %3.2f " %( f), end='')
        # print("")
    except:
            traceback.print_exc();

def Get_Retimer0Temp():
    """
        Read Retimer Temp and provide output in tabulated format
    """
    try:
        with urllib.request.urlopen("http://192.168.31.1/redfish/v1/Chassis/RETIMER_0/Sensors/RETIMER_0_TEMP") as url:
            j= json.load(url)
            # print(j)
            print(json.dumps(j, indent=4))
    except:
            traceback.print_exc();


def Get_EventLogs():
    """
        Read /redfish/v1/Systems/UBB/LogServices/EventLog/Entries and provide output in tabulated format
        Get all EventLogs
        python3 /tmp/GT_bmc_pyutil.py -c  Get_EventLogs
        python3 /tmp/GT_bmc_pyutil.py -c  Get_EventLogs_Sorted
    """
    sURL = "http://192.168.31.1/redfish/v1/Systems/UBB/LogServices/EventLog/Entries"
    print("sURL = %s" %(sURL)) 
    with urllib.request.urlopen(sURL) as url:
        j= json.load(url)
        # print(j)
        # print(json.dumps(j, indent=4))
        i=0
        Members = j['Members']
        for m in Members:
            # Id, Severity, Created, Message
            print("%s - %-10s - %s ---  %s" %(m['Id'], m['Severity'],m['Created'],m['Message']));
        print("Count:", j['Members@odata.count']);
        while 'Members@odata.nextLink' in j:
            print("NextLink:", j['Members@odata.nextLink']);
            nextLink = j['Members@odata.nextLink'];
            sURL = "http://192.168.31.1" + nextLink
            print("   sURL = %s" %(sURL)) 
            with urllib.request.urlopen(sURL) as u2:
                j = json.load(u2)
                Members = j['Members']
                for m in Members:
                    # Id, Severity, Created, Message
                    print("%s - %-10s - %s ---  %s" %(m['Id'], m['Severity'],m['Created'],m['Message']));
                print("Count:", j['Members@odata.count']);
                if ('Members@odata.nextLink' in j): 
                    print("NextLink:", j['Members@odata.nextLink']);
                        

def Get_EventLogs_Sorted():
    """
        Read /redfish/v1/Systems/UBB/LogServices/EventLog/Entries and provide output in tabulated format
        Get all EventLogs
        python3 /tmp/GT_bmc_pyutil.py -c  Get_EventLogs_Sorted
    """
    sURL = "http://192.168.31.1/redfish/v1/Systems/UBB/LogServices/EventLog/Entries"
    print("sURL = %s" %(sURL)) 
    
    with urllib.request.urlopen(sURL) as url:
        j= json.load(url)
        # print(j)
        # print(json.dumps(j, indent=4))
        i=0
        Members = j['Members']
        oEV_Entries = Members
        #for m in Members:
        #    # Id, Severity, Created, Message
        #    print("%s - %-10s - %s ---  %s" %(m['Id'], m['Severity'],m['Created'],m['Message']));
        print("Count:", j['Members@odata.count']);
        while 'Members@odata.nextLink' in j:
            print("NextLink:", j['Members@odata.nextLink']);
            nextLink = j['Members@odata.nextLink'];
            sURL = "http://192.168.31.1" + nextLink
            print("   sURL = %s" %(sURL)) 
            with urllib.request.urlopen(sURL) as u2:
                j = json.load(u2)
                Members = j['Members']
                oEV_Entries.extend(Members)
                # for m in Members:
                #    # Id, Severity, Created, Message
                #    print("%s - %-10s - %s ---  %s" %(m['Id'], m['Severity'],m['Created'],m['Message']));
                print("Count:", j['Members@odata.count']);
                if ('Members@odata.nextLink' in j): 
                    print("NextLink:", j['Members@odata.nextLink']);

        oEV_Entries_sorted = sorted(oEV_Entries, key=lambda d: int(d['Id']))
        print("Members@odata.count  :", j['Members@odata.count']);
        print("oEV_Entries Count    :", len(oEV_Entries_sorted));

        nID = int(oEV_Entries_sorted[0]['Id'])-1
        for m in oEV_Entries_sorted: 
            nCurID = int(m['Id'])
            if ((nID+1)!= nCurID):
                print("Warning ID %d is out of order" %(nCurID))
            print("%s - %-10s - %s ---  %s" %(m['Id'], m['Severity'],m['Created'],m['Message']));
            nID = nCurID

def Get_Telemetries():
    with urllib.request.urlopen("http://192.168.31.1/redfish/v1/TelemetryService/MetricReports/All") as url:
        j= json.load(url)
        for m in  j['MetricValues']:
            print ("%-60s -- %20s   %s " %(m['MetricProperty'][19:-8], m['MetricValue'], m['Timestamp']));

def Get_TelemetriesRetimers():
    """
        Read /redfish/v1/TelemetryService/MetricReports/All and provide output in tabulated format
    """
    with urllib.request.urlopen("http://192.168.31.1/redfish/v1/TelemetryService/MetricReports/All") as url:
        j= json.load(url)
        # print(j)
        # print(json.dumps(j, indent=4))
        MetricValues = j['MetricValues']
        for i in range(0,8):
            s = "/redfish/v1/Chassis/RETIMER_%d/Sensors/RETIMER_%d_TEMP/Reading" %(i, i)
            print(s);
            for m in MetricValues:
                if (m['MetricProperty']==s):    
                    print (m['MetricValue'], m['Timestamp']);


def DecodeEventLogFromStdIn():
    """
        Parse json output of EventLog from stdin
        print it out in table format
        C_GETJ  /redfish/v1/Systems/UBB/LogServices/EventLog/Entries | python3 /tmp/parse_EventLog.py
    """
    j = json.load(sys.stdin)
    # print(j['Members'])
    # print(j['Members'][0], len(j['Members']))
    for i in range(len(j['Members'])): 
        # Id, Created, Message, Severity
        m = j['Members'][i]
        print("%-6s - %s - %-10s ---   %s" %(m['Id'], m['Created'], m['Severity'], m['Message']))


if __name__ == "__main__":
    main()
