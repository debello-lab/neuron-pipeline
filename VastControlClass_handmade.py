import socket
import struct
import numpy as np
from typing import List, Tuple, Any, Optional, Dict

ERROR_CODES = {
            0: "No error",
            1: "Unknown error",
            2: "Unexpected data received from VAST - API mismatch?",
            3: "VAST received invalid data. Command ignored.",
            4: "VAST internal data read failure",
            5: "Internal VAST error",
            6: "Could not complete command because modifying the view in VAST is disabled",
            7: "Could not complete command because modifying the segmentation in VAST is disabled",
            10: "Coordinates out of bounds",
            11: "Mip level out of bounds",
            12: "Data size overflow",
            13: "Data size mismatch - Specified coordinates and data block size do not match",
            14: "Parameter out of bounds",
            15: "Could not enable diskcache (please set correct folder in VAST Preferences)",
            20: "Annotation object or segment number out of bounds",
            21: "No annotation or segmentation available",
            22: "RLE overflow - RLE-encoding makes the data larger than raw; please request as raw",
            23: "Invalid annotation object type",
            24: "Annotation operation failed",
            30: "Invalid layer number",
            31: "Invalid layer type",
            32: "Layer operation failed",
            50: "sourcearray and targetarray must have the same length",
        }

class VASTControlClass:

        GETINFO = 1
    GETNUMBEROFSEGMENTS = 2
    GETSEGMENTDATA = 3
    GETSEGMENTNAME = 4
    SETANCHORPOINT = 5
    SETSEGMENTNAME = 6
    SETSEGMENTCOLOR = 7
    GETVIEWCOORDINATES = 8
    GETVIEWZOOM = 9
    SETVIEWCOORDINATES = 10
    SETVIEWZOOM = 11
    GETNROFLAYERS = 12
    GETLAYERINFO = 13
    GETALLSEGMENTDATA = 14
    GETALLSEGMENTNAMES = 15
    SETSELECTEDSEGMENTNR = 16
    GETSELECTEDSEGMENTNR = 17
    SETSELECTEDLAYERNR = 18
    GETSELECTEDLAYERNR = 19
    GETSEGIMAGERAW = 20
    GETSEGIMAGERLE = 21
    GETSEGIMAGESURFRLE = 22
    SETSEGTRANSLATION = 23
    GETSEGIMAGERAWIMMEDIATE = 24
    GETSEGIMAGERLEIMMEDIATE = 25
    GETEMIMAGERAW = 30
    GETEMIMAGERAWIMMEDIATE = 31
    REFRESHLAYERREGION = 32
    GETPIXELVALUE = 33
    GETSCREENSHOTIMAGERAW = 40
    GETSCREENSHOTIMAGERLE = 41
    SETSEGIMAGERAW = 50
    SETSEGIMAGERLE = 51
    SETSEGMENTBBOX = 60
    GETFIRSTSEGMENTNR = 61
    GETHARDWAREINFO = 62
    ADDSEGMENT = 63
    MOVESEGMENT = 64
    GETDRAWINGPROPERTIES = 65
    SETDRAWINGPROPERTIES = 66
    GETFILLINGPROPERTIES = 67
    SETFILLINGPROPERTIES = 68

    GETANNOLAYERNROFOBJECTS = 70
    GETANNOLAYEROBJECTDATA = 71
    GETANNOLAYEROBJECTNAMES = 72
    ADDNEWANNOOBJECT = 73
    MOVEANNOOBJECT = 74
    REMOVEANNOOBJECT = 75
    SETSELECTEDANNOOBJECTNR = 76
    GETSELECTEDANNOOBJECTNR = 77
    GETAONODEDATA = 78
    GETAONODELABELS = 79

    SETSELECTEDAONODEBYDFSNR = 80
    SETSELECTEDAONODEBYCOORDS = 81
    GETSELECTEDAONODENR = 82
    ADDAONODE = 83
    MOVESELECTEDAONODE = 84
    REMOVESELECTEDAONODE = 85
    SWAPSELECTEDAONODECHILDREN = 86
    MAKESELECTEDAONODEROOT = 87

    SPLITSELECTEDSKELETON = 88
    WELDSKELETONS = 89

    GETANNOOBJECT = 90
    SETANNOOBJECT = 91
    ADDANNOOBJECT = 92
    GETCLOSESTAONODEBYCOORDS = 93
    GETAONODEPARAMS = 94
    SETAONODEPARAMS = 95

    GETAPIVERSION = 100
    GETAPILAYERSENABLED = 101
    SETAPILAYERSENABLED = 102
    GETSELECTEDAPILAYERNR = 103
    SETSELECTEDAPILAYERNR = 104

    GETCURRENTUISTATE = 110
    GETERRORPOPUPSENABLED = 112
    SETERRORPOPUPSENABLED = 113
    SETUIMODE = 114
    SHOWWINDOW = 115
    SET2DVIEWORIENTATION = 116
    GETPOPUPSENABLED = 117
    SETPOPUPSENABLED = 118

    ADDNEWLAYER = 120
    LOADLAYER = 121
    SAVELAYER = 122
    REMOVELAYER = 123
    MOVELAYER = 124
    SETLAYERINFO = 125
    GETMIPMAPSCALEFACTORS = 126

    EXECUTEFILL = 131
    EXECUTELIMITEDFILL = 132
    EXECUTECANVASPAINTSTROKE = 133
    EXECUTESTARTAUTOSKELETONIZATION = 134
    EXECUTESTOPAUTOSKELETONIZATION = 135
    EXECUTEISAUTOSKELETONIZATIONDONE = 136

    SETTOOLPARAMETERS = 151

    def __init__(self):
        self.isconnected      = 0;
        self.inres            = [];
        self.nrinints         = 0;
        self.inintdata        = [];
        self.nrinuints        = 0;
        self.inuintdata       = [];
        self.nrindoubles      = 0;
        self.indoubledata     = [];
        self.nrinchars        = 0;
        self.inchardata       = [];
        self.nrintext         = 0;
        self.intextdata       = {};
        self.nrinuint64s      = 0;
        self.inuint64data     = [];
        self.parseheaderok    = 0;
        self.parseheaderlen   = 0;
        self.lasterror        = 0;
        self.thisversionnr    = 5; #Version 5 is for VAST Lite 1.3, 1.4, 1.5
        self.thissubversionnr = 14;
        self.indata           = [];
        self.host             = "127.0.0.1"
        self.port             = 22081
        self.client_socket    = None

    def connect(self, host, port, timeout):
        self.host = host
        self.port = port
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(timeout)
            self.client_socket.connect((self.host, self.port))
            self.isconnected = 1
        except Exception as e:
            self.isconnected = 0
            print(f"Connection error: {e}")
        return self.isconnected

    def disconnect(self):
        if self.client_socket is not None:
            try:
                self.client_socket.close()
                self.isconnected = 0
            except Exception as e:
                print(f"Disconnection error: {e}")
        
        return 1

    def getlasterror(self):
        return self.lasterror

    ######################################################################
    # GETINFO

    def getinfo(self) -> Optional[Dict[str, Any]]:
    
