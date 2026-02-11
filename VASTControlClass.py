import socket
import struct
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

HOST = "127.0.0.1"
PORT = 22081

# VAST API message IDs
GETINFO = 1;
GETNUMBEROFSEGMENTS = 2;
GETSEGMENTDATA     = 3;
GETSEGMENTNAME     = 4;
SETANCHORPOINT     = 5;
SETSEGMENTNAME     = 6;
SETSEGMENTCOLOR    = 7;
GETVIEWCOORDINATES = 8;
GETVIEWZOOM        = 9;
SETVIEWCOORDINATES = 10;
SETVIEWZOOM        = 11;
GETNROFLAYERS      = 12;
GETLAYERINFO       = 13;
GETALLSEGMENTDATA  = 14;
GETALLSEGMENTNAMES = 15;
SETSELECTEDSEGMENTNR = 16;
GETSELECTEDSEGMENTNR = 17;
SETSELECTEDLAYERNR = 18;
GETSELECTEDLAYERNR = 19;
GETSEGIMAGERAW     = 20;
GETSEGIMAGERLE     = 21;
GETSEGIMAGESURFRLE = 22;
SETSEGTRANSLATION  = 23;
GETSEGIMAGERAWIMMEDIATE = 24;
GETSEGIMAGERLEIMMEDIATE = 25;
GETEMIMAGERAW      = 30;
GETEMIMAGERAWIMMEDIATE = 31;
REFRESHLAYERREGION = 32;
GETPIXELVALUE      = 33;
GETSCREENSHOTIMAGERAW = 40;
GETSCREENSHOTIMAGERLE = 41;
SETSEGIMAGERAW     = 50;
SETSEGIMAGERLE     = 51;
SETSEGMENTBBOX     = 60;
GETFIRSTSEGMENTNR  = 61;
GETHARDWAREINFO    = 62;
ADDSEGMENT         = 63;
MOVESEGMENT        = 64;
GETDRAWINGPROPERTIES = 65;
SETDRAWINGPROPERTIES = 66;
GETFILLINGPROPERTIES = 67;
SETFILLINGPROPERTIES = 68;

GETANNOLAYERNROFOBJECTS = 70;
GETANNOLAYEROBJECTDATA  = 71;
GETANNOLAYEROBJECTNAMES  = 72;
ADDNEWANNOOBJECT = 73;
MOVEANNOOBJECT = 74;
REMOVEANNOOBJECT = 75;
SETSELECTEDANNOOBJECTNR = 76;
GETSELECTEDANNOOBJECTNR = 77;
GETAONODEDATA = 78;
GETAONODELABELS = 79;

SETSELECTEDAONODEBYDFSNR = 80;
SETSELECTEDAONODEBYCOORDS = 81;
GETSELECTEDAONODENR = 82;
ADDAONODE           = 83;
MOVESELECTEDAONODE          = 84;
REMOVESELECTEDAONODE        = 85;
SWAPSELECTEDAONODECHILDREN  = 86;
MAKESELECTEDAONODEROOT      = 87;

SPLITSELECTEDSKELETON = 88;
WELDSKELETONS = 89;

GETANNOOBJECT = 90;
SETANNOOBJECT = 91;
ADDANNOOBJECT = 92;
GETCLOSESTAONODEBYCOORDS = 93;
GETAONODEPARAMS = 94;
SETAONODEPARAMS = 95;

GETAPIVERSION      = 100;
GETAPILAYERSENABLED = 101;
SETAPILAYERSENABLED = 102;
GETSELECTEDAPILAYERNR = 103;
SETSELECTEDAPILAYERNR = 104;

GETCURRENTUISTATE  = 110;
GETERRORPOPUPSENABLED = 112;
SETERRORPOPUPSENABLED = 113;
SETUIMODE          = 114;
SHOWWINDOW         = 115;
SET2DVIEWORIENTATION = 116;
GETPOPUPSENABLED = 117;
SETPOPUPSENABLED = 118;

ADDNEWLAYER        = 120;
LOADLAYER          = 121;
SAVELAYER          = 122;
REMOVELAYER        = 123;
MOVELAYER          = 124;
SETLAYERINFO       = 125;
GETMIPMAPSCALEFACTORS = 126;

EXECUTEFILL        = 131;
EXECUTELIMITEDFILL = 132;
EXECUTECANVASPAINTSTROKE = 133;
EXECUTESTARTAUTOSKELETONIZATION=134;
EXECUTESTOPAUTOSKELETONIZATION=135;
EXECUTEISAUTOSKELETONIZATIONDONE=136;

SETTOOLPARAMETERS  = 151;

errorCodes = {
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
    def __init__(self, host=HOST, port=PORT):
        self.host                            = host
        self.port                            = port
        self.client: Optional[socket.socket] = None
        self.last_error                      = 0
    
    #########################
    ## FUNDAMENTAL METHODS ##
    #########################

    def connect(self, timeout=100):
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(timeout)
            self.client.connect((self.host, self.port))
            return 1
        except Exception as e:
            print(f"Connection failed: {e}")
            try:
                if self.client:
                    self.client.close()
            except Exception:
                self.last_error = 1
                pass
            self.client = None
            return 0
    
    def disconnect(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
        return 1
    
    def _encode_text(self, text: str) -> bytes:
        """
        Encode text as type 3 (null-terminated string).
        Format: [3][text bytes][0]
        """
        text_bytes = text.encode('utf-8')
        return b'\x03' + text_bytes + b'\x00'

    @staticmethod
    def _encode_uint32(value: int) -> bytes:
        """
        Equivalent of MATLAB bytesfromuint32(uint32(value)).
        Format (from parse.m tags):
        [1][4 bytes of uint32 little-endian]
        """
        return b"\x01" + struct.pack("<I", value & 0xFFFFFFFF)

    @staticmethod
    def _encode_int32(value: int) -> bytes:
        """bytesfromint32(int32(value)) | for other APIs later."""
        return b"\x04" + struct.pack("<i", value)
        
    @staticmethod
    def _encode_double(value: float) -> bytes:
        # tag 2 + float64 LE
        return b"\x02" + struct.pack("<d", float(value))
    
    def _decode_to_struct(self, indata: bytes) -> Tuple[Dict[str, Any], int]:
        """
        Decode binary data into a structured dictionary.
        
        Converts binary-encoded data with variable names and types into a Python dictionary.
        Format: [name\0][type][value]... where type determines how to parse the value.
        
        Type codes:
        0: uint32 (4 bytes)
        1: int32 (4 bytes)
        2: uint64 (8 bytes)
        3: double (8 bytes)
        4: string (null-terminated)
        5: uint32 matrix (xsize, ysize, data)
        6: int32 matrix (xsize, ysize, data)
        7: uint64 matrix (xsize, ysize, data)
        8: double matrix (xsize, ysize, data)
        9: string array/cell array
        
        Returns: (out_dict, success) where success=1 if parsing was successful, 0 otherwise
        """
        out = {}
        ptr = 0
        data_len = len(indata)
        
        try:
            while ptr < data_len:
                # Get variable name (null-terminated string)
                ptr2 = ptr
                while ptr2 < data_len and indata[ptr2] != 0:
                    ptr2 += 1
                
                if ptr2 >= data_len:
                    break
                
                variable_name = indata[ptr:ptr2].decode('utf-8', errors='replace')
                ptr = ptr2 + 1
                
                if ptr >= data_len:
                    break
                
                # Get type
                data_type = indata[ptr]
                ptr += 1
                
                if data_type == 0:  # uint32
                    if ptr + 4 > data_len:
                        break
                    value = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    out[variable_name] = int(value)
                    ptr += 4
                
                elif data_type == 1:  # int32
                    if ptr + 4 > data_len:
                        break
                    value = struct.unpack("<i", indata[ptr:ptr+4])[0]
                    out[variable_name] = int(value)
                    ptr += 4
                
                elif data_type == 2:  # uint64
                    if ptr + 8 > data_len:
                        break
                    value = struct.unpack("<Q", indata[ptr:ptr+8])[0]
                    out[variable_name] = int(value)
                    ptr += 8
                
                elif data_type == 3:  # double
                    if ptr + 8 > data_len:
                        break
                    value = struct.unpack("<d", indata[ptr:ptr+8])[0]
                    out[variable_name] = value
                    ptr += 8
                
                elif data_type == 4:  # character string
                    ptr2 = ptr
                    while ptr2 < data_len and indata[ptr2] != 0:
                        ptr2 += 1
                    
                    if ptr2 >= data_len:
                        break
                    
                    value = indata[ptr:ptr2].decode('utf-8', errors='replace')
                    out[variable_name] = value
                    ptr = ptr2 + 1
                
                elif data_type == 5:  # uint32 matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    
                    matrix_bytes = xsize * ysize * 4
                    if ptr + matrix_bytes > data_len:
                        break
                    
                    mtx_data = struct.unpack(f"<{xsize*ysize}I", indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.uint32).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes
                
                elif data_type == 6:  # int32 matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    
                    matrix_bytes = xsize * ysize * 4
                    if ptr + matrix_bytes > data_len:
                        break
                    
                    mtx_data = struct.unpack(f"<{xsize*ysize}i", indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.int32).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes
                
                elif data_type == 7:  # uint64 matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    
                    matrix_bytes = xsize * ysize * 8
                    if ptr + matrix_bytes > data_len:
                        break
                    
                    mtx_data = struct.unpack(f"<{xsize*ysize}Q", indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.uint64).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes
                
                elif data_type == 8:  # double matrix
                    if ptr + 8 > data_len:
                        break
                    xsize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    ysize = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    
                    matrix_bytes = xsize * ysize * 8
                    if ptr + matrix_bytes > data_len:
                        break
                    
                    mtx_data = struct.unpack(f"<{xsize*ysize}d", indata[ptr:ptr+matrix_bytes])
                    mtx = np.array(mtx_data, dtype=np.float64).reshape((ysize, xsize))
                    out[variable_name] = mtx
                    ptr += matrix_bytes
                
                elif data_type == 9:  # array of strings (cell array)
                    if ptr + 8 > data_len:
                        break
                    
                    nrofstrings = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    totaltextlength = struct.unpack("<I", indata[ptr:ptr+4])[0]
                    ptr += 4
                    
                    if ptr + totaltextlength > data_len:
                        break
                    
                    str_data = indata[ptr:ptr+totaltextlength]
                    ptr += totaltextlength
                    
                    if nrofstrings > 0:
                        ca = []
                        p = 0
                        for i in range(nrofstrings):
                            p2 = p
                            while p2 < len(str_data) and str_data[p2] != 0:
                                p2 += 1
                            
                            if p2 < len(str_data):
                                ca.append(str_data[p:p2].decode('utf-8', errors='replace'))
                                p = p2 + 1
                        
                        out[variable_name] = ca
                
                else:
                    break  # Unknown type
            
            return out, 1
        
        except Exception as e:
            return {}, 0
    
    def _encode_from_struct(self, indata: Dict[str, Any]) -> Tuple[bytes, int]:
        """
        Encode a Python dictionary into binary structured format.
        
        Converts a Python dictionary into binary format with variable names and types.
        Format: [name\0][type][value]... where type indicates the data type.
        
        Supported Python types map to:
        - np.uint32 or int (scalar) -> type 0
        - np.int32 or int (scalar) -> type 1
        - int (large) or np.int64 -> type 2
        - float -> type 3
        - str (scalar) -> type 4
        - np.ndarray(uint32) -> type 5
        - np.ndarray(int32) -> type 6
        - np.ndarray(uint64) -> type 7
        - np.ndarray(float64) -> type 8
        - list/tuple (strings) -> type 9
        
        Returns: (bytes, success) where success=1 if encoding was successful, 0 otherwise
        """
        try:
            if not isinstance(indata, dict):
                return b"", 0
            
            out = bytearray()
            
            for key, value in indata.items():
                key_bytes = key.encode('utf-8')
                
                # Handle scalar and single-value fields
                if isinstance(value, int) and not isinstance(value, (list, np.ndarray)):
                    # Check range to determine int type
                    if value > 0xFFFFFFFF or value < -0x80000000:
                        # int64/uint64
                        out.extend(key_bytes + b'\x00\x02' + struct.pack("<Q", int(value) & 0xFFFFFFFFFFFFFFFF))
                    elif value >= 0 and value <= 0xFFFFFFFF:
                        # uint32
                        out.extend(key_bytes + b'\x00\x00' + struct.pack("<I", int(value)))
                    elif value >= -2147483648 and value <= 2147483647:
                        # int32
                        out.extend(key_bytes + b'\x00\x01' + struct.pack("<i", int(value)))
                
                elif isinstance(value, float):
                    # double
                    out.extend(key_bytes + b'\x00\x03' + struct.pack("<d", value))
                
                elif isinstance(value, str):
                    # character string
                    str_bytes = value.encode('utf-8')
                    out.extend(key_bytes + b'\x00\x04' + str_bytes + b'\x00')
                
                elif isinstance(value, np.ndarray):
                    # Matrix
                    mtx = value.T if value.ndim == 2 else np.atleast_1d(value).T
                    xsize = np.uint32(mtx.shape[0])
                    ysize = np.uint32(mtx.shape[1])
                    
                    if value.dtype == np.uint32:
                        out.extend(key_bytes + b'\x00\x05')
                        out.extend(struct.pack("<I", xsize) + struct.pack("<I", ysize))
                        out.extend(mtx.astype(np.uint32).tobytes())
                    
                    elif value.dtype == np.int32:
                        out.extend(key_bytes + b'\x00\x06')
                        out.extend(struct.pack("<I", xsize) + struct.pack("<I", ysize))
                        out.extend(mtx.astype(np.int32).tobytes())
                    
                    elif value.dtype == np.uint64:
                        out.extend(key_bytes + b'\x00\x07')
                        out.extend(struct.pack("<I", xsize) + struct.pack("<I", ysize))
                        out.extend(mtx.astype(np.uint64).tobytes())
                    
                    elif value.dtype == np.float64 or value.dtype == float:
                        out.extend(key_bytes + b'\x00\x08')
                        out.extend(struct.pack("<I", xsize) + struct.pack("<I", ysize))
                        out.extend(mtx.astype(np.float64).tobytes())
                
                elif isinstance(value, (list, tuple)):
                    # Array of strings (cell array)
                    cstr = bytearray()
                    nrofstrings = np.uint32(len(value))
                    
                    for s in value:
                        s_bytes = str(s).encode('utf-8')
                        cstr.extend(s_bytes + b'\x00')
                    
                    totaltextlength = np.uint32(len(cstr))
                    out.extend(key_bytes + b'\x00\x09')
                    out.extend(struct.pack("<I", nrofstrings))
                    out.extend(struct.pack("<I", totaltextlength))
                    out.extend(cstr)
            
            return bytes(out), 1
        
        except Exception as e:
            return b"", 0
    
    def send_command(self, msg_id: int, payload: bytes = b"") -> Tuple[int, bytes]:
        """
        Send a binary command to the VAST API and return the message type and response bytes.

        Each message follows the format:
        [0..3]   = b"VAST"
        [4..11]  = uint64 length of data following (payload + 4 bytes for msg_id)
        [12..15] = uint32 message ID (little-endian)
        [16..]   = payload (optional)
        """
        client = self.client
        if client is None:
            raise RuntimeError("Not connected to VAST API.")
        
        total_len = len(payload) + 4
        header    = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
        message   = header + payload

        # print(f"Sending {len(message)} bytes: {message.hex()}")

        
        client.sendall(message)

        # Receive response header 
        hdr = client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            raise RuntimeError(f"Invalid response header: {hdr!r}")

        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type  = struct.unpack("<I", hdr[12:16])[0]

        expected = total_len - 4
        payload_bytes = bytearray()
        while len(payload_bytes) < expected:
            chunk = client.recv(expected - len(payload_bytes))
            if not chunk:
                raise RuntimeError("Incomplete payload from VAST.")
            payload_bytes.extend(chunk)

        # print(f"Response msg_type={msg_type}, len={len(payload_bytes)}")
        return msg_type, bytes(payload_bytes)

    def parse_payload(self, indata: bytes) -> Dict[str, Any]:
        """
        Python equivalent of the inner part of MATLAB parse(obj, indata).

        indata is assumed to be ONLY the typed payload (no 'VAST' header), returned by self.send_command().

        Returns a dict:
            {
                "ints":    [int32, ...],
                "uints":   [uint32, ...],
                "doubles": [float64, ...],
                "text":    [str, ...],      # all 0-terminated text chunks
                "last_text": str | "",      # last inchardata
                "uint64s": [int, ...],      # uint64 values
            }
        """
        ints: List[int]      = []
        uints: List[int]     = []
        doubles: List[float] = []
        texts: List[str]     = []
        last_text: str       = ""
        uint64s: List[int]   = []

        p = 0
        n = len(indata)
        while p < n:
            t   = indata[p]

            if t   == 1:  # uint32
                if p + 5 > n:
                    break
                val = struct.unpack("<I", indata[p+1:p+5])[0]
                uints.append(val)
                p  += 5
            elif t == 2: # double | float64
                if p + 9 > n:
                    break
                val = struct.unpack("<d", indata[p+1:p+9])[0]
                doubles.append(val)
                p  += 9
            elif t == 3: # 0-terminated text
                q = p + 1
                while q < n and indata[q] != 0:
                    q += 1
                if q  >= n or indata[q]   != 0:
                    break # abort parsing
                text_bytes = indata[p+1:q]
                try:
                    s = text_bytes.decode("utf-8", errors="replace")
                except Exception:
                    s = "".join(chr(b) for b in text_bytes)
                last_text = s
                texts.append(s)
                p = q + 1 # skip null terminator
            elif t == 4: # int32
                if p + 5 > n:
                    break
                val = struct.unpack("<i", indata[p+1:p+5])[0]
                ints.append(val)
                p  += 5
            elif t == 6: # uint64
                if p + 9 > n:
                    break
                val = struct.unpack("<Q", indata[p+1:p+9])[0]
                uint64s.append(val)
                p  += 9
            else:
                break # unknown type
        return {
            "ints":    ints,
            "uints":   uints,
            "doubles":  doubles,
            "text":  texts,
            "last_text": last_text,
            "uint64s":   uint64s,
        }
    
    ############################
    #     GENERAL FUNCTIONS    #
    ############################

    def get_info(self) -> dict: 
        """
        Get general information from VAST.
        
        Returns dict with dataset info, or empty dict on failure.
        Fields: datasizex/y/z, voxelsizex/y/z, cubesizex/y/z, 
                currentviewx/y/z, nrofmiplevels
        """
        msg_type, data = self.send_command(GETINFO)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        doubles = parsed.get("doubles", [])
        ints = parsed.get("ints", [])
        
        if len(uints) == 7 and len(doubles) == 3 and len(ints) == 3:
            self.last_error = 0
            return {
                "datasizex":     uints[0],
                "datasizey":     uints[1],
                "datasizez":     uints[2],
                "voxelsizex":    doubles[0],
                "voxelsizey":    doubles[1],
                "voxelsizez":    doubles[2],
                "cubesizex":     uints[3],
                "cubesizey":     uints[4],
                "cubesizez":     uints[5],
                "currentviewx":  ints[0],
                "currentviewy":  ints[1],
                "currentviewz":  ints[2],
                "nrofmiplevels": uints[6],
            }
        else:
            self.last_error = 2
            return {}
    
    def get_api_version(self) -> int:
        msg_type, data = self.send_command(GETAPIVERSION)
        if msg_type == 21:
            self.last_error = 21
            print(f"Error getting API version: {errorCodes[msg_type]}")
            return 0
        try:
            version = struct.unpack("<I", data[1:5])[0]
            return version
        except struct.error as e:
            self.last_error = 2
            print(f"Failed to parse version response: {data!r}, error: {e}")
            return 0

    def get_hardware_info(self) -> dict:
        msg_type, data = self.send_command(GETHARDWAREINFO)
        if msg_type == 21:
            self.last_error = 21
            print(f"Error getting hardware info: {errorCodes[msg_type]}")
            return {}
        try:
            parsed = self.parse_payload(data)

            u32    = parsed.get("uints", [])
            f64    = parsed.get("doubles", [])
            i32    = parsed.get("ints", [])
            texts  = parsed.get("text", [])

            # Expected: 1 uint, 7 doubles, 0 ints, 5 text strings
            if len(u32) != 1 or len(f64) != 7 or len(i32) != 0 or len(texts) != 5:
                print(
                    f"Unexpected payload layout: "
                    f"uints={len(u32)}, doubles={len(f64)}, ints={len(i32)}, text={len(texts)}"
                )
                return {}
            
            info = {
                "computername":                texts[0],
                "processorname":               texts[1],
                "processorspeed_ghz":          f64[0],
                "nrofprocessorcores":          u32[0],
                "tickspeedmhz":                f64[1],
                "mmxssecapabilities":          texts[2],
                "totalmemorygb":               f64[2],
                "freememorygb":                f64[3],
                "graphicscardname":            texts[3],
                "graphicsdedicatedvideomemgb": f64[4],
                "graphicsdedicatedsysmemgb":   f64[5],
                "graphicssharedsysmemgb":      f64[6],
                "graphicsrasterizerused":      texts[4],
            }

            return info
        except ValueError as e:
            self.last_error = 2
            print(f"Failed to parse hardware info response: {data!r}, error: {e}")
            return {}

    def get_last_error(self) -> int:
        """Query VAST for the most recent error code (like MATLAB getlasterror)."""
        return self.last_error

    def set_error_popups_enabled(self, code, enabled: bool) -> bool:
        """
        Enable or disable error popups in VAST for a specific error code.
        Corresponds to MATLAB seterrorpopupenabled(code, enabled).
        """
        payload = self._encode_uint32(code) + self._encode_uint32(1 if enabled else 0)
        msg_type, data = self.send_command(SETERRORPOPUPSENABLED, payload)

        if msg_type == 1:
            self.last_error = 0
            return True
        else:
            self.last_error = msg_type
            print(f"Error setting error popup enabled: {errorCodes.get(msg_type, 'Unknown error')}")
            return False
        
    ############################
    #     LAYER FUNCTIONS      #
    ############################

    def get_number_of_layers(self) -> int:
        """ Get the number of layers in the current VAST session. """
        msg_type, data = self.send_command(GETNROFLAYERS)
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])

        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            print(f"Unexpected payload structure: uint32={len(uints)}")
            return 0

    def get_layer_info(self, layer_nr: int) -> dict:
        """
        Retrieve information about a specific layer in VAST.
        Corresponds to MATLAB getlayerinfo(layernr).
        """
        payload = self._encode_uint32(layer_nr)
        msg_type, data = self.send_command(GETLAYERINFO, payload)

        parsed  = self.parse_payload(data)
        ints    = parsed.get("ints", [])
        uints   = parsed.get("uints", [])
        doubles = parsed.get("doubles", [])
        name    = parsed.get("last_text", "")

        if not (len(ints) == 8 and len(uints) == 7 and len(doubles) == 3 and name):
            self.last_error = 2
            print(f"Unexpected payload structure: ints={len(ints)}, uints={len(uints)}, doubles={len(doubles)}, text={len(name)}")
            return {}

        self.last_error = 0
        
        layer_type_map = {
            0: "Image",
            1: "Segmentation",
            2: "Annotation",
            3: "VSVR image",
            6: "VSVI image",
            7: "Tool",
        }
        layer_type = layer_type_map.get(ints[0], "Other")
        layerinfo = {
            "type":             layer_type,
            "editable":         ints[1],
            "visible":          ints[2],
            "brightness":       ints[3],
            "contrast":         ints[4],

            "opacitylevel":     doubles[0],
            "brightnesslevel":  doubles[1],
            "contrastlevel":    doubles[2],

            "blendmode":        ints[5],
            "blendoradd":       ints[6],

            "tintcolor":        uints[0],
            "name":             name,

            "redtargetcolor":   uints[1],
            "greentargetcolor": uints[2],
            "bluetargetcolor":  uints[3],
            "bytesperpixel":    uints[4],

            "ischanged":        ints[7],
            "inverted":         uints[5],
            "solomode":         uints[6],
        }

        return layerinfo        

    def set_layer_info(self, layer_nr: int, layer_info: dict) -> bool:
        """
        Set information for a specific layer in VAST.
        Corresponds to MATLAB setlayerinfo(layernr, layerinfo).
        """
        if not layer_info:
            self.last_error = 0
            return True
        xflags = 0
        msg = self._encode_uint32(layer_nr)

        def has(key: str) -> bool:
            return key in layer_info and layer_info[key] is not None
        
        # 1  editable (int32)
        if has("editable"):
            msg += self._encode_int32(layer_info["editable"])
            xflags |= 1

        # 2  visible (int32)
        if has("visible"):
            msg += self._encode_int32(layer_info["visible"])
            xflags |= 2

        # 4  brightness (int32)
        if has("brightness"):
            msg += self._encode_int32(layer_info["brightness"])
            xflags |= 4

        # 8  contrast (int32)
        if has("contrast"):
            msg += self._encode_int32(layer_info["contrast"])
            xflags |= 8

        # 16 opacitylevel (double)
        if has("opacitylevel"):
            msg += self._encode_double(layer_info["opacitylevel"])
            xflags |= 16

        # 32 brightnesslevel (double)
        if has("brightnesslevel"):
            msg += self._encode_double(layer_info["brightnesslevel"])
            xflags |= 32

        # 64 contrastlevel (double)
        if has("contrastlevel"):
            msg += self._encode_double(layer_info["contrastlevel"])
            xflags |= 64

        # 128 blendmode (int32)
        if has("blendmode"):
            msg += self._encode_int32(layer_info["blendmode"])
            xflags |= 128

        # 256 blendoradd (int32)
        if has("blendoradd"):
            msg += self._encode_int32(layer_info["blendoradd"])
            xflags |= 256

        # 512 tintcolor (uint32)
        if has("tintcolor"):
            msg += self._encode_uint32(layer_info["tintcolor"])
            xflags |= 512

        # 1024 redtargetcolor (uint32)
        if has("redtargetcolor"):
            msg += self._encode_uint32(layer_info["redtargetcolor"])
            xflags |= 1024

        # 2048 greentargetcolor (uint32)
        if has("greentargetcolor"):
            msg += self._encode_uint32(layer_info["greentargetcolor"])
            xflags |= 2048

        # 4096 bluetargetcolor (uint32)
        if has("bluetargetcolor"):
            msg += self._encode_uint32(layer_info["bluetargetcolor"])
            xflags |= 4096

        # 8192 inverted (uint32)
        if has("inverted"):
            msg += self._encode_uint32(layer_info["inverted"])
            xflags |= 8192

        # 16384 solomode (uint32)
        if has("solomode"):
            msg += self._encode_uint32(layer_info["solomode"])
            xflags |= 16384

        # If after all that xflags is still 0, nothing is set → no-op
        if xflags == 0:
            self.last_error = 0
            return True

        # Final payload: [bytesfromuint32(xflags), msg...]
        payload = self._encode_uint32(xflags) + msg

        # Send command and look at error code in header
        msg_type, data = self.send_command(SETLAYERINFO, payload)

        success = (msg_type == 1)
        if success:
            self.last_error = 0
        else:
            # Optional: we could parse 'data' to find an error code or
            # add a get_last_error() call here. For now, just flag nonzero.
            self.last_error = -1

        return success

    def get_data_size_at_mip(self, layer_nr: int, mip: int) -> Optional[tuple]:
        """
        Get the data size at a specific mip level for a given layer in VAST.
        Corresponds to MATLAB getdatasizeatmip(layernr, miplevel).
        """
        xyz_size = []
        info = self.get_info()
        if not info:
            return None
        
        msf = self.get_mipmap_scale_factors(layer_nr)

        if mip >= 0 and mip < info['nrofmiplevels']:
                if mip == 0:
                    # Full resolution
                    xyzsize = (
                        float(info['datasizex']),
                        float(info['datasizey']),
                        float(info['datasizez'])
                    )
                else:
                    # Scaled resolution
                    if not msf or mip > len(msf):
                        return None
                    
                    xyzsize = (
                        int(float(info['datasizex']) / msf[mip - 1][0]),
                        int(float(info['datasizey']) / msf[mip - 1][1]),
                        int(float(info['datasizez']) / msf[mip - 1][2])
                    )
                
                return xyzsize
            
        return None

    def get_mipmap_scale_factors(self, layer_nr: int) -> List[List[int]]:
        """
        Get the mipmap scale factors for a specific layer in VAST.
        Corresponds to MATLAB getmipmapscalefactors(layernr).
        """
        payload = self._encode_uint32(layer_nr)

        msg_type, raw = self.send_command(GETMIPMAPSCALEFACTORS, payload)

        if len(raw) < 4 or (len(raw) % 4) != 0:
            # At least one uint32 needed, and payload must be multiple of 4 bytes
            self.last_error = 2  # "unexpected data received"
            return []

        # Interpret payload as an array of uint32 (little-endian)
        count = len(raw) // 4
        uid = struct.unpack("<{}I".format(count), raw)

        num_levels = uid[0]  # same as uid(1) in MATLAB

        # We expect 1 + 3*num_levels entries total
        if 1 + 3 * num_levels > len(uid):
            self.last_error = 2
            return []

        # Build matrix including mip level 0
        matrix: List[List[int]] = []
        idx = 1  # start from uid(2) in MATLAB (0-based index 1)

        for _ in range(num_levels):
            sx = uid[idx]
            sy = uid[idx + 1]
            sz = uid[idx + 2]
            matrix.append([sx, sy, sz])
            idx += 3

        # Drop first row: mip level 0 (always [1,1,1])
        if matrix:
            matrix = matrix[1:]

        self.last_error = 0
        return matrix

    def set_selected_layer_nr(self, layer_nr: int) -> bool:
        """
        Set the selected layer in VAST.
        Corresponds to MATLAB setselectedlayernr(layernr).
        """
        payload = self._encode_uint32(layer_nr)
        msg_type, data = self.send_command(SETSELECTEDLAYERNR, payload)
        
        success = (msg_type == 1)
        if success:
            self.last_error = 0
        else:
            self.last_error = -1

        return success
    
    def get_selected_layer_nr(self) -> dict:
        """
        Get the selected layer number in VAST.
        Corresponds to MATLAB getselectedlayernr().
        """
        msg_type, data = self.send_command(GETSELECTEDLAYERNR)

        # msg_type is 'res' (1 = success, 0 = failure), not an error code
        if msg_type != 1:
            self.last_error = -1  
            return {}

        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])

        # MATLAB cases:
        #   if nrinints == 3:
        #       [sel, sel_em, sel_seg] = inintdata(1..3)
        #   elseif nrinints == 5:
        #       sel     = inintdata(1)
        #       sel_em  = inintdata(2)
        #       sel_seg = inintdata(4)
        if len(ints) == 3:
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[2]
        elif len(ints) == 5:
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[3]
        else:
            # Mirror MATLAB: unexpected data received
            self.last_error = 2  # "Unexpected data received from VAST - API mismatch?"
            return {}

        # If we got here, treat as success; -1 is a *valid* "none" value.
        self.last_error = 0
        return {
            "selected_layer": selected_layer,
            "selected_em_layer": selected_em_layer,
            "selected_segment_layer": selected_segment_layer,
        }
    
    def get_selected_layer_nr2(self) -> dict:
        msg_type, data = self.send_command(GETSELECTEDLAYERNR)

        # msg_type is 'res': 1 = success, 0 = failure
        if msg_type != 1:
            self.last_error = -1  # generic failure; can refine with getlasterror later
            return {}

        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])

        # Defaults as in MATLAB
        selected_layer = -1
        selected_em_layer = -1
        selected_anno_layer = -1
        selected_segment_layer = -1
        selected_tool_layer = -1

        if len(ints) == 3:
            # Backwards compatibility case
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[2]
            # anno/tool remain -1
        elif len(ints) == 5:
            selected_layer        = ints[0]
            selected_em_layer     = ints[1]
            selected_anno_layer   = ints[2]
            selected_segment_layer= ints[3]
            selected_tool_layer   = ints[4]
        else:
            # Unexpected layout
            self.last_error = 2  # "unexpected data received"
            return {}

        self.last_error = 0
        return {
            "selected_layer":         selected_layer,
            "selected_em_layer":      selected_em_layer,
            "selected_anno_layer":    selected_anno_layer,
            "selected_segment_layer": selected_segment_layer,
            "selected_tool_layer":    selected_tool_layer,
        }

    def add_new_layer(self, layer_type: int, name: str, ref_id: int = -1) -> int:
        """Add a new layer to VAST."""
        payload = self._encode_int32(layer_type) + self._encode_int32(ref_id) + \
                self._encode_text(name)
        
        msg_type, data = self.send_command(ADDNEWLAYER, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def load_layer(self, filename: str, ref_id: int = -1) -> int:
        """Load a layer from file."""
        payload = self._encode_int32(ref_id) + self._encode_text(filename)
        
        msg_type, data = self.send_command(LOADLAYER, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def save_layer(self, layer_nr: int, filename: str, force: bool = False, 
                subformat: int = 0) -> bool:
        """Save a layer to file."""
        payload = self._encode_uint32(layer_nr) + \
                self._encode_uint32(int(force)) + \
                self._encode_uint32(subformat) + \
                self._encode_text(filename)
        
        msg_type, data = self.send_command(SAVELAYER, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success
    
    def remove_layer(self, layer_nr: int, force: bool = False) -> bool:
        """
        Remove a layer.
        
        Args:
            layer_nr: Layer number to remove
            force: Force removal without confirmation
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(layer_nr) + self._encode_uint32(int(force))
        msg_type, data = self.send_command(REMOVELAYER, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def move_layer(self, moved_layer_nr: int, after_layer_nr: int) -> int:
        """
        Move a layer to a new position in the layer list.
        
        Args:
            moved_layer_nr: Layer to move
            after_layer_nr: Insert after this layer
        
        Returns:
            New layer number, or 0 on failure
        """
        payload = self._encode_uint32(moved_layer_nr) + \
                self._encode_uint32(after_layer_nr)
        
        msg_type, data = self.send_command(MOVELAYER, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def get_api_layers_enabled(self) -> int:
        """
        Check if separate API layer selection is enabled.
        
        Returns:
            1 if enabled, 0 if disabled, -1 on error
        """
        msg_type, data = self.send_command(GETAPILAYERSENABLED)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return -1

    def set_api_layers_enabled(self, enabled: bool) -> bool:
        """
        Enable or disable separate API layer selection.
        
        When enabled, API functions use layers set via set_selected_api_layer_nr().
        When disabled, API functions use the layer selected in VAST UI.
        
        Args:
            enabled: True to enable API layer selection, False to disable
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(int(enabled))
        msg_type, data = self.send_command(SETAPILAYERSENABLED, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_api_layer_nr(self) -> dict:
        """
        Get the currently selected API layer numbers.
        
        Returns dict with keys:
            selected_layer: Most recently selected layer of any type
            selected_em_layer: Selected EM/image layer
            selected_anno_layer: Selected annotation layer
            selected_segment_layer: Selected segmentation layer
            selected_tool_layer: Selected tool layer
        
        Returns empty dict on failure.
        """
        msg_type, data = self.send_command(GETSELECTEDAPILAYERNR)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])
        
        if len(ints) == 5:
            self.last_error = 0
            return {
                "selected_layer": ints[0],
                "selected_em_layer": ints[1],
                "selected_anno_layer": ints[2],
                "selected_segment_layer": ints[3],
                "selected_tool_layer": ints[4],
            }
        else:
            self.last_error = 2
            return {}

    def set_selected_api_layer_nr(self, layer_nr: int) -> bool:
        """
        Select a layer for API access.
        
        Note: API layer control must be enabled via set_api_layers_enabled(True)
        for this selection to be used by API functions.
        
        Args:
            layer_nr: Layer number to select
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_int32(layer_nr)
        msg_type, data = self.send_command(SETSELECTEDAPILAYERNR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    ############################
    #   ANNOTATION FUNCTIONS   #
    ############################

    def get_anno_object(self, object_id: int) -> dict:
        """
        Get complete annotation object data as structured dict.
        
        Args:
            object_id: Annotation object ID (0 = currently selected)
        
        Returns:
            Dict with all object data, or empty dict on failure
        """
        payload = self._encode_uint32(object_id)
        msg_type, data = self.send_command(GETANNOOBJECT, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        # Decode structured data
        obj_data, success = self._decode_to_struct(data)
        
        if success != 1:
            self.last_error = 2
            return {}
        
        self.last_error = 0
        return obj_data

    def get_ao_node_params(self, anno_object_id: int, node_dfsnr: int) -> dict:
        """
        Get detailed parameters for a specific node.
        
        Args:
            anno_object_id: Annotation object ID (0 = currently selected)
            node_dfsnr: Node DFS number
        
        Returns:
            Dict with node parameters, or empty dict on failure
        """
        payload = self._encode_uint32(anno_object_id) + \
                self._encode_uint32(node_dfsnr)
        
        msg_type, data = self.send_command(GETAONODEPARAMS, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        # Decode structured data
        node_data, success = self._decode_to_struct(data)
        
        if success != 1:
            self.last_error = 2
            return {}
        
        self.last_error = 0
        return node_data

    def set_anno_object(self, object_id: int, obj_data: dict) -> int:
        """
        Set/update annotation object data.
        
        Args:
            object_id: Object ID to update
            obj_data: Dict with object data (from get_anno_object format)
        
        Returns:
            Object ID on success, 0 on failure
        """
        encoded_data, success = self._encode_from_struct(obj_data)
        payload = self._encode_uint32(object_id) + encoded_data
        
        msg_type, data = self.send_command(SETANNOOBJECT, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def add_anno_object(self, ref_id: int, next_or_child: int, obj_data: dict) -> int:
        """
        Add new annotation object with complete data.
        
        Args:
            ref_id: Reference object ID
            next_or_child: 0=next sibling, 1=child
            obj_data: Dict with object data
        
        Returns:
            New object ID, or 0 on failure
        """
        encoded_data, success = self._encode_from_struct(obj_data)
        payload = self._encode_uint32(ref_id) + \
                self._encode_uint32(next_or_child) + \
                encoded_data
        
        msg_type, data = self.send_command(ADDANNOOBJECT, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def set_ao_node_params(self, anno_object_id: int, node_dfsnr: int, 
                        node_data: dict) -> bool:
        """
        Set parameters for a specific node.
        
        Args:
            anno_object_id: Annotation object ID
            node_dfsnr: Node DFS number
            node_data: Dict with node parameters
        
        Returns:
            True on success, False on failure
        """
        encoded_data, success = self._encode_from_struct(node_data)
        payload = self._encode_uint32(anno_object_id) + \
                self._encode_uint32(node_dfsnr) + \
                encoded_data
        
        msg_type, data = self.send_command(SETAONODEPARAMS, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_anno_layer_nr_of_objects(self) -> tuple:
        """Get number of annotation objects and first object number."""
        msg_type, data = self.send_command(GETANNOLAYERNROFOBJECTS)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0, -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 2:
            self.last_error = 0
            return uints[0], uints[1]
        else:
            self.last_error = 2
            return 0, -1

    def get_ao_node_data(self) -> Optional[np.ndarray]:
        """
        Get all skeleton node data from selected annotation layer.
        
        Returns numpy array with columns:
        0: node index (DFS number)
        1: isselected flag
        2: edge flags
        3: has label flag
        4: reserved
        5-10: parent, child1, child2, prev, next, nrofchildren (uint32, -1 means none)
        11: radius (double)
        12-13: x, y coordinates (uint32)
        """
        msg_type, data = self.send_command(GETAONODEDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        
        # Parse as uint32 array
        arr = np.frombuffer(data, dtype=np.uint32)
        
        if len(arr) < 1:
            self.last_error = 2
            return None
        
        num_nodes = arr[0]
        node_data = np.zeros((num_nodes, 14))
        
        sp = 1
        for i in range(num_nodes):
            node_data[i, 0] = i  # DFS number
            node_data[i, 1] = arr[sp] & 0xFF  # isselected
            node_data[i, 2] = (arr[sp] >> 8) & 0xFF  # edgeflags
            node_data[i, 3] = (arr[sp] >> 16) & 0xFF  # haslabel
            node_data[i, 4] = (arr[sp] >> 24) & 0xFF  # reserved
            
            # Hierarchy (convert 0xFFFFFFFF to -1)
            for j in range(6):
                val = arr[sp + 1 + j]
                node_data[i, 5 + j] = -1 if val == 0xFFFFFFFF else val
            
            # Radius (double at position sp+7, sp+8)
            radius_bytes = data[4 * (sp + 7):4 * (sp + 9)]
            node_data[i, 11] = np.frombuffer(radius_bytes, dtype=np.float64)[0]
            
            # X, Y coordinates
            node_data[i, 12] = arr[sp + 9]
            node_data[i, 13] = arr[sp + 10]
            
            sp += 11
        
        self.last_error = 0
        return node_data

    def get_anno_layer_object_data(self) -> List[dict]:
        """Get metadata for all annotation objects."""
        msg_type, data = self.send_command(GETANNOLAYEROBJECTDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return []
        
        arr = np.frombuffer(data, dtype=np.uint32)
        
        if len(arr) < 1:
            return []
        
        num_objects = arr[0]
        objects = []
        
        sp = 1
        for i in range(num_objects):
            obj = {
                "id": i + 1,
                "type": arr[sp] & 0xFFFF,  # 0=folder, 1=skeleton
                "flags": (arr[sp] >> 16) & 0xFFFF,
                "col1": arr[sp + 1],
                "col2": arr[sp + 2],
                "anchorpoint": [arr[sp + 3], arr[sp + 4], arr[sp + 5]],
                "hierarchy": [arr[sp + 6], arr[sp + 7], arr[sp + 8], arr[sp + 9]],
                "collapsednr": arr[sp + 10],
                "boundingbox": [arr[sp + 11], arr[sp + 12], arr[sp + 13], 
                            arr[sp + 14], arr[sp + 15], arr[sp + 16]],
            }
            objects.append(obj)
            sp += 21
        
        self.last_error = 0
        return objects

    def set_selected_anno_object_nr(self, object_id: int) -> bool:
        """Select an annotation object by ID."""
        payload = self._encode_uint32(object_id)
        msg_type, data = self.send_command(SETSELECTEDANNOOBJECTNR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_anno_layer_object_names(self) -> List[str]:
        """
        Get names of all annotation objects in the selected annotation layer.
        
        Returns:
            List of object names, or empty list on failure
        """
        msg_type, data = self.send_command(GETANNOLAYEROBJECTNAMES)
        
        if msg_type != 1:
            self.last_error = msg_type
            return []
        
        if len(data) < 4:
            self.last_error = 2
            return []
        
        
        # First uint32 is number of names
        num_names = struct.unpack("<I", data[0:4])[0]
        
        names = []
        pos = 4
        
        for i in range(num_names):
            # Find null terminator
            end_pos = pos
            while end_pos < len(data) and data[end_pos] != 0:
                end_pos += 1
            
            if end_pos >= len(data):
                self.last_error = 2
                return []
            
            # Extract name
            name_bytes = data[pos:end_pos]
            try:
                name = name_bytes.decode('utf-8', errors='replace')
            except:
                name = ''.join(chr(b) for b in name_bytes)
            
            names.append(name)
            pos = end_pos + 1  # Skip null terminator
        
        self.last_error = 0
        return names

    def add_new_anno_object(self, ref_id: int, next_or_child: int, 
                            obj_type: int, name: str) -> int:
        """
        Add a new annotation object.
        
        Args:
            ref_id: Reference object ID
            next_or_child: 0=add as next sibling, 1=add as child
            obj_type: 0=folder, 1=skeleton
            name: Name for the new object
        
        Returns:
            ID of new object, or 0 on failure
        """
        payload = self._encode_uint32(ref_id) + \
                self._encode_uint32(next_or_child) + \
                self._encode_uint32(obj_type) + \
                self._encode_text(name)
        
        msg_type, data = self.send_command(ADDNEWANNOOBJECT, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def move_anno_object(self, obj_id: int, ref_id: int, next_or_child: int) -> bool:
        """
        Move an annotation object in the hierarchy.
        
        Args:
            obj_id: Object to move
            ref_id: Reference object
            next_or_child: 0=move as next sibling, 1=move as child
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(obj_id) + \
                self._encode_uint32(ref_id) + \
                self._encode_uint32(next_or_child)
        
        msg_type, data = self.send_command(MOVEANNOOBJECT, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def remove_anno_object(self, obj_id: int) -> bool:
        """
        Remove an annotation object.
        
        Args:
            obj_id: Object ID to remove
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(obj_id)
        msg_type, data = self.send_command(REMOVEANNOOBJECT, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_anno_object_nr(self) -> int:
        """
        Get the currently selected annotation object number.
        
        Returns:
            Object ID, or -1 on failure
        """
        msg_type, data = self.send_command(GETSELECTEDANNOOBJECTNR)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return -1

    def get_ao_node_labels(self) -> tuple:
        """
        Get labels for skeleton nodes that have labels.
        
        Returns:
            Tuple of (node_numbers, labels) where:
            - node_numbers: List[int] of node DFS numbers
            - labels: List[str] of corresponding labels
            Returns ([], []) on failure
        """
        msg_type, data = self.send_command(GETAONODELABELS)
        
        if msg_type != 1:
            self.last_error = msg_type
            return [], []
        
        if len(data) < 4:
            self.last_error = 2
            return [], []
        
        
        # Number of labels
        num_labels = struct.unpack("<I", data[0:4])[0]
        
        if num_labels == 0:
            self.last_error = 0
            return [], []
        
        # Node numbers (uint32 array)
        node_data = data[4:4 + num_labels * 4]
        node_numbers = list(np.frombuffer(node_data, dtype=np.uint32))
        
        # Extract label strings
        labels = []
        pos = 4 + num_labels * 4
        
        for i in range(num_labels):
            # Find null terminator
            end_pos = pos
            while end_pos < len(data) and data[end_pos] != 0:
                end_pos += 1
            
            if end_pos >= len(data):
                self.last_error = 2
                return [], []
            
            # Extract label
            label_bytes = data[pos:end_pos]
            try:
                label = label_bytes.decode('utf-8', errors='replace')
            except:
                label = ''.join(chr(b) for b in label_bytes)
            
            labels.append(label)
            pos = end_pos + 1
        
        self.last_error = 0
        return node_numbers, labels

    def set_selected_ao_node_by_dfsnr(self, node_dfsnr: int) -> bool:
        """
        Select a skeleton node by its depth-first search number.
        
        Args:
            node_dfsnr: DFS number of node to select
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(node_dfsnr)
        msg_type, data = self.send_command(SETSELECTEDAONODEBYDFSNR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def set_selected_ao_node_by_coords(self, x: int, y: int, z: int) -> bool:
        """
        Select the skeleton node closest to the given coordinates.
        
        Args:
            x, y, z: Coordinates in voxels (full resolution)
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(x) + self._encode_uint32(y) + \
                self._encode_uint32(z)
        
        msg_type, data = self.send_command(SETSELECTEDAONODEBYCOORDS, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_ao_node_nr(self) -> int:
        """
        Get the DFS number of the currently selected skeleton node.
        
        Returns:
            Node DFS number, or -1 on failure
        """
        msg_type, data = self.send_command(GETSELECTEDAONODENR)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return -1

    def add_ao_node(self, x: int, y: int, z: int) -> bool:
        """
        Add a new node to the selected skeleton.
        
        Args:
            x, y, z: Coordinates for new node (full resolution voxels)
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(x) + self._encode_uint32(y) + \
                self._encode_uint32(z)
        
        msg_type, data = self.send_command(ADDAONODE, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def move_selected_ao_node(self, x: int, y: int, z: int) -> bool:
        """
        Move the selected skeleton node to new coordinates.
        
        Args:
            x, y, z: New coordinates (full resolution voxels)
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(x) + self._encode_uint32(y) + \
                self._encode_uint32(z)
        
        msg_type, data = self.send_command(MOVESELECTEDAONODE, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def remove_selected_ao_node(self) -> bool:
        """
        Remove the currently selected skeleton node.
        
        Returns:
            True on success, False on failure
        """
        msg_type, data = self.send_command(REMOVESELECTEDAONODE)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def swap_selected_ao_node_children(self) -> bool:
        """
        Swap the two children of the selected skeleton node.
        
        Returns:
            True on success, False on failure
        """
        msg_type, data = self.send_command(SWAPSELECTEDAONODECHILDREN)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def make_selected_ao_node_root(self) -> bool:
        """
        Make the selected node the root of the skeleton tree.
        
        Returns:
            True on success, False on failure
        """
        msg_type, data = self.send_command(MAKESELECTEDAONODEROOT)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def split_selected_skeleton(self, new_root_dfsnr: int, new_name: str) -> int:
        """
        Split skeleton by removing parent edge and creating new skeleton.
        
        Args:
            new_root_dfsnr: DFS number of node to become root of new skeleton
            new_name: Name for the new skeleton object
        
        Returns:
            ID of new skeleton object, or 0 on failure
        """
        payload = self._encode_uint32(new_root_dfsnr) + self._encode_text(new_name)
        
        msg_type, data = self.send_command(SPLITSELECTEDSKELETON, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def weld_skeletons(self, anno_object_nr1: int, node_dfsnr1: int,
                   anno_object_nr2: int, node_dfsnr2: int) -> bool:
        """
        Weld two skeletons together at specified nodes.
        
        Replaces node1 with node2, appends node1's parent to node2's parent,
        and removes node1's parent edge.
        
        Args:
            anno_object_nr1: First skeleton object ID
            node_dfsnr1: Node DFS number in first skeleton
            anno_object_nr2: Second skeleton object ID
            node_dfsnr2: Node DFS number in second skeleton
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(anno_object_nr1) + \
                self._encode_uint32(node_dfsnr1) + \
                self._encode_uint32(anno_object_nr2) + \
                self._encode_uint32(node_dfsnr2)
        
        msg_type, data = self.send_command(WELDSKELETONS, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_closest_ao_node_by_coords(self, x: int, y: int, z: int, 
                                  max_distance: int) -> tuple:
        """
        Find the closest skeleton node to given coordinates.
        
        Args:
            x, y, z: Query coordinates (full resolution voxels)
            max_distance: Maximum search distance
        
        Returns:
            Tuple of (anno_object_id, node_dfsnr, distance) or (-1, -1, -1) on failure
        """
        payload = self._encode_uint32(x) + self._encode_uint32(y) + \
                self._encode_uint32(z) + self._encode_uint32(max_distance)
        
        msg_type, data = self.send_command(GETCLOSESTAONODEBYCOORDS, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1, -1, -1
        
        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])
        doubles = parsed.get("doubles", [])
        
        if len(ints) == 2 and len(doubles) == 1:
            self.last_error = 0
            return ints[0], ints[1], doubles[0]
        else:
            self.last_error = 0
            return -1, -1, -1

    ############################
    #   SEGMENTATION FUNCTIONS #
    ############################

    def get_number_of_segments(self) -> int:
        """Get the number of segments in the current segmentation."""
        msg_type, data = self.send_command(GETNUMBEROFSEGMENTS)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def get_segment_data(self, segment_id: int) -> dict:
        """
        Get metadata for a specific segment.
        
        Returns dict with:
            id, flags, col1, col2, anchorpoint, hierarchy, collapsednr, boundingbox
        """
        payload = self._encode_uint32(segment_id)
        msg_type, data = self.send_command(GETSEGMENTDATA, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])
        uints = parsed.get("uints", [])
        
        if len(ints) == 9 and len(uints) == 9:
            self.last_error = 0
            return {
                "id": uints[0],
                "flags": uints[1],
                "col1": uints[2],
                "col2": uints[3],
                "anchorpoint": ints[0:3],
                "hierarchy": list(uints[4:8]),
                "collapsednr": uints[8],
                "boundingbox": ints[3:9],
            }
        else:
            self.last_error = 2
            return {}

    def get_segment_name(self, segment_id: int) -> str:
        """Get the name of a specific segment."""
        payload = self._encode_uint32(segment_id)
        msg_type, data = self.send_command(GETSEGMENTNAME, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return ""
        
        parsed = self.parse_payload(data)
        name = parsed.get("last_text", "")
        
        if name:
            self.last_error = 0
            return name
        else:
            self.last_error = 2
            return ""

    def set_anchor_point(self, segment_id: int, x: int, y: int, z: int) -> bool:
        """
        Set the anchor point of a segment.
        
        Args:
            segment_id: Segment ID
            x, y, z: Anchor coordinates in voxels (full resolution, non-negative)
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(segment_id) + \
                self._encode_uint32(x) + \
                self._encode_uint32(y) + \
                self._encode_uint32(z)
        
        msg_type, data = self.send_command(SETANCHORPOINT, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def set_segment_name(self, segment_id: int, name: str) -> bool:
        """
        Set the name of a segment.
        
        Args:
            segment_id: Segment ID
            name: New name for the segment
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(segment_id) + self._encode_text(name)
        
        msg_type, data = self.send_command(SETSEGMENTNAME, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def set_segment_color_8(self, segment_id: int, r1: int, g1: int, b1: int, p1: int,
                            r2: int, g2: int, b2: int, p2: int) -> bool:
        """
        Set segment colors using 8-bit RGB values.
        
        Args:
            segment_id: Segment ID
            r1, g1, b1: Primary color (0-255)
            p1: Pattern (0-15)
            r2, g2, b2: Secondary color (0-255)
            p2: Secondary pattern (currently unused)
        
        Returns:
            True on success, False on failure
        """
        # Pack colors into 32-bit values: [pattern][blue][green][red]
        v1 = (p1 & 0xFF) | ((b1 & 0xFF) << 8) | ((g1 & 0xFF) << 16) | ((r1 & 0xFF) << 24)
        v2 = (p2 & 0xFF) | ((b2 & 0xFF) << 8) | ((g2 & 0xFF) << 16) | ((r2 & 0xFF) << 24)
        
        payload = self._encode_uint32(segment_id) + \
                self._encode_uint32(v1) + \
                self._encode_uint32(v2)
        
        msg_type, data = self.send_command(SETSEGMENTCOLOR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def set_segment_color_32(self, segment_id: int, col1: int, col2: int) -> bool:
        """
        Set segment colors using 32-bit color values.
        
        Args:
            segment_id: Segment ID
            col1: Primary color as 32-bit value
            col2: Secondary color as 32-bit value
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(segment_id) + \
                self._encode_uint32(col1) + \
                self._encode_uint32(col2)
        
        msg_type, data = self.send_command(SETSEGMENTCOLOR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_all_segment_data(self) -> List[dict]:
        """
        Get metadata for all segments.
        
        Returns list of dicts, each containing segment data.
        """
        msg_type, data = self.send_command(GETALLSEGMENTDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return []
        
        # For large data, might need chunked reading
        if len(data) < 4:
            self.last_error = 2
            return []
        
        # Parse as uint32 array
        arr = np.frombuffer(data, dtype=np.uint32)
        iarr = np.frombuffer(data, dtype=np.int32)
        
        num_segments = arr[0]
        segments = []
        
        sp = 1  # start position (0-based in Python)
        for i in range(num_segments):
            seg = {
                "id": i,
                "flags": arr[sp],
                "col1": arr[sp + 1],
                "col2": arr[sp + 2],
                "anchorpoint": [int(iarr[sp + 3]), int(iarr[sp + 4]), int(iarr[sp + 5])],
                "hierarchy": [int(arr[sp + 6]), int(arr[sp + 7]), int(arr[sp + 8]), int(arr[sp + 9])],
                "collapsednr": arr[sp + 10],
                "boundingbox": [int(iarr[sp + 11]), int(iarr[sp + 12]), int(iarr[sp + 13]),
                               int(iarr[sp + 14]), int(iarr[sp + 15]), int(iarr[sp + 16])],
            }
            segments.append(seg)
            sp += 17
        
        self.last_error = 0
        return segments

    def get_all_segment_data_matrix(self) -> Tuple[np.ndarray, int]:
        """
        Get metadata for all segments as a matrix.        
        Columns are:
        0: segment index (0-based)
        1: flags
        2-5: primary color RGB + pattern (8-bit values extracted from col1)
        6-9: secondary color RGB + pattern (8-bit values extracted from col2)
        10-12: anchorpoint (x, y, z)
        13-16: hierarchy (parent, child1, child2, next)
        17: collapsednr
        18-23: boundingbox (minx, maxx, miny, maxy, minz, maxz)
        
        Returns:
            Tuple of (matrix, success) where matrix is numpy array of shape (num_segments-1, 24)
            or empty array on failure, and success is 1 or 0
        """
        
        msg_type, data = self.send_command(GETALLSEGMENTDATA)
        
        if msg_type != 1:
            self.last_error = msg_type
            return np.array([]), 0
        
        # Check minimum data size (need at least count field)
        if len(data) < 4:
            self.last_error = 2
            return np.array([]), 0
        
        try:
            # Parse as uint32 and int32 arrays
            # MATLAB code: uid=typecast(obj.indata(17:end), 'uint32');
            # In MATLAB, obj.indata(17:end) skips 16 bytes of header
            # But in Python, send_command already stripped the header, so we start at 0
            uid = np.frombuffer(data, dtype=np.uint32)
            sid = np.frombuffer(data, dtype=np.int32)
            
            # First uint32 is the count (MATLAB: uid(1))
            # In MATLAB arrays are 1-indexed, Python is 0-indexed
            num_segments = uid[0]
            
            if num_segments == 0:
                self.last_error = 0
                return np.zeros((0, 24), dtype=np.float64), 1
            
            # Verify we have enough data
            # Each segment needs 17 uint32 values
            expected_uint32_count = 1 + num_segments * 17
            if len(uid) < expected_uint32_count:
                self.last_error = 2
                return np.array([]), 0
            
            # Create output matrix: num_segments x 24 columns
            segdatamatrix = np.zeros((num_segments, 24), dtype=np.float64)
            
            # MATLAB starts at sp=2 (1-indexed, so element 2 is uid(2))
            # In Python 0-indexed, this is sp=1
            sp = 1
            
            # MATLAB loop: for i=1:1:uid(1)
            for i in range(num_segments):
                # Column 0: segment index (MATLAB: segdatamatrix(i,1)=i-1)
                segdatamatrix[i, 0] = i
                
                # Column 1: flags (MATLAB: segdatamatrix(i,2)=uid(sp))
                segdatamatrix[i, 1] = uid[sp]
                
                # Columns 2-5: Extract RGBP from col1
                # MATLAB: segdatamatrix(i,3)=bitand(bitshift(uid(sp+1),-24),255)
                segdatamatrix[i, 2] = (uid[sp + 1] >> 24) & 0xFF  # R
                segdatamatrix[i, 3] = (uid[sp + 1] >> 16) & 0xFF  # G
                segdatamatrix[i, 4] = (uid[sp + 1] >> 8) & 0xFF   # B
                segdatamatrix[i, 5] = uid[sp + 1] & 0xFF          # P
                
                # Columns 6-9: Extract RGBP from col2
                # MATLAB: segdatamatrix(i,7)=bitand(bitshift(uid(sp+2),-24),255)
                segdatamatrix[i, 6] = (uid[sp + 2] >> 24) & 0xFF  # R
                segdatamatrix[i, 7] = (uid[sp + 2] >> 16) & 0xFF  # G
                segdatamatrix[i, 8] = (uid[sp + 2] >> 8) & 0xFF   # B
                segdatamatrix[i, 9] = uid[sp + 2] & 0xFF          # P
                
                # Columns 10-12: anchorpoint (x, y, z)
                # MATLAB: segdatamatrix(i,11:13)=id(sp+3:sp+5)
                # MATLAB sp+3:sp+5 is inclusive, so 3 elements starting at sp+3
                # In Python, this is sp+3:sp+6 (MATLAB sp+3:sp+5 = 3 elements)
                segdatamatrix[i, 10:13] = sid[sp + 3:sp + 6]
                
                # Columns 13-16: hierarchy (parent, child1, child2, next)
                # MATLAB: segdatamatrix(i,14:17)=uid(sp+6:sp+9)
                # MATLAB sp+6:sp+9 is inclusive, so 4 elements
                # In Python, this is sp+6:sp+10 (MATLAB sp+6:sp+9 = 4 elements)
                segdatamatrix[i, 13:17] = uid[sp + 6:sp + 10]
                
                # Column 17: collapsednr
                # MATLAB: segdatamatrix(i,18)=uid(sp+10)
                segdatamatrix[i, 17] = uid[sp + 10]
                
                # Columns 18-23: boundingbox (minx, maxx, miny, maxy, minz, maxz)
                # MATLAB: segdatamatrix(i,19:24)=id(sp+11:sp+16)
                # MATLAB sp+11:sp+16 is inclusive, so 6 elements
                # In Python, this is sp+11:sp+17 (MATLAB sp+11:sp+16 = 6 elements)
                segdatamatrix[i, 18:24] = sid[sp + 11:sp + 17]
                
                # Move to next segment
                sp += 17
            
            # Remove first row (segment at index 0, which is typically background/invalid)
            # MATLAB: segdatamatrix=segdatamatrix(2:end,:)
            # MATLAB 2:end means skip first row (index 1 in MATLAB)
            # In Python, this is [1:, :]
            if segdatamatrix.shape[0] > 0:
                segdatamatrix = segdatamatrix[1:, :]
            
            self.last_error = 0
            return segdatamatrix, 1
        
        except Exception as e:
            print(f"Error in get_all_segment_data_matrix: {e}")
            self.last_error = 2
            return np.array([]), 0

    def get_all_segment_names(self) -> List[str]:
        """Get names of all segments in the dataset."""
        msg_type, data = self.send_command(GETALLSEGMENTNAMES)
        
        if msg_type != 1:
            self.last_error = msg_type
            return []
        
        if len(data) < 4:
            self.last_error = 2
            return []
        
        
        # First uint32 is number of names
        num_names = struct.unpack("<I", data[0:4])[0]
        
        names = []
        pos = 4
        
        for i in range(num_names):
            # Find null terminator
            end_pos = pos
            while end_pos < len(data) and data[end_pos] != 0:
                end_pos += 1
            
            if end_pos >= len(data):
                self.last_error = 2
                return []
            
            # Extract name
            name_bytes = data[pos:end_pos]
            try:
                name = name_bytes.decode('utf-8', errors='replace')
            except:
                name = ''.join(chr(b) for b in name_bytes)
            
            names.append(name)
            pos = end_pos + 1  # Skip null terminator
        
        self.last_error = 0
        return names  

    def set_selected_segment_nr(self, segment_id: int) -> bool:
        """
        Select a segment.
        
        Args:
            segment_id: Segment ID to select
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_int32(segment_id)
        msg_type, data = self.send_command(SETSELECTEDSEGMENTNR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_segment_nr(self) -> int:
        """
        Get the currently selected segment number.
        
        Returns:
            Segment ID, or -1 on failure
        """
        msg_type, data = self.send_command(GETSELECTEDSEGMENTNR)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return -1

    def set_segment_bbox(self, segment_id: int, minx: int, maxx: int,
                     miny: int, maxy: int, minz: int, maxz: int) -> bool:
        """
        Set the bounding box of a segment.
        
        Args:
            segment_id: Segment ID
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(segment_id) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(SETSEGMENTBBOX, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_first_segment_nr(self) -> int:
        """
        Get the ID of the first segment in the hierarchy.
        
        Returns:
            Segment ID, or -1 on failure
        """
        msg_type, data = self.send_command(GETFIRSTSEGMENTNR)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return -1
    
    def add_segment(self, ref_id: int, next_or_child: int, name: str) -> int:
        """
        Add a new segment to the hierarchy.
        
        Args:
            ref_id: Reference segment ID
            next_or_child: 0=add as next sibling, 1=add as child
            name: Name for new segment
        
        Returns:
            New segment ID, or 0 on failure
        """
        payload = self._encode_uint32(ref_id) + \
                self._encode_uint32(next_or_child) + \
                self._encode_text(name)
        
        msg_type, data = self.send_command(ADDSEGMENT, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return 0

    def move_segment(self, segment_id: int, ref_id: int, next_or_child: int) -> bool:
        """
        Move a segment in the hierarchy.
        
        Args:
            segment_id: Segment to move
            ref_id: Reference segment
            next_or_child: 0=move as next sibling, 1=move as child
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(segment_id) + \
                self._encode_uint32(ref_id) + \
                self._encode_uint32(next_or_child)
        
        msg_type, data = self.send_command(MOVESEGMENT, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    #############################
    #     2D VIEW FUNCTIONS     #
    #############################

    def get_view_coordinates(self) -> tuple:
        """
        Get current view coordinates in VAST.
        
        Returns:
            Tuple of (x, y, z) in pixels at mip0, or (0, 0, 0) on failure
        """
        msg_type, data = self.send_command(GETVIEWCOORDINATES)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0, 0, 0
        
        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])
        
        if len(ints) == 3:
            self.last_error = 0
            return ints[0], ints[1], ints[2]
        else:
            self.last_error = 2
            return 0, 0, 0

    def set_view_coordinates(self, x: int, y: int, z: int) -> bool:
        """
        Set view coordinates in VAST (navigate to position).
        
        Args:
            x, y, z: Coordinates in pixels at mip0
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(x) + self._encode_uint32(y) + \
                self._encode_uint32(z)
        
        msg_type, data = self.send_command(SETVIEWCOORDINATES, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_view_zoom(self) -> int:
        """
        Get current zoom level (mip level).
        
        Returns:
            Zoom level, or 0 on failure
        """
        msg_type, data = self.send_command(GETVIEWZOOM)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        ints = parsed.get("ints", [])
        
        if len(ints) == 1:
            self.last_error = 0
            return ints[0]
        else:
            self.last_error = 2
            return 0

    def set_view_zoom(self, zoom: int) -> bool:
        """
        Set zoom level (mip level).
        
        Args:
            zoom: Mip level to set
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_int32(zoom)
        msg_type, data = self.send_command(SETVIEWZOOM, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def refresh_layer_region(self, layer_nr: int, minx: int, maxx: int,
                         miny: int, maxy: int, minz: int, maxz: int) -> bool:
        """
        Refresh a region of a layer in the VAST display.
        
        Args:
            layer_nr: Layer number to refresh
            minx, maxx, miny, maxy, minz, maxz: Region to refresh
        
        Returns:
            True on success, False on failure
        """
        payload = self._encode_uint32(layer_nr) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(REFRESHLAYERREGION, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success
    
    # set2dvieworientation not needed

    #################################
    # Voxel Data Transfer Functions #
    #################################        

    def get_seg_image_raw(self, miplevel: int, minx: int, maxx: int, 
                          miny: int, maxy: int, minz: int, maxz: int,
                          immediate: bool = False, request_load: bool = False) -> Optional[bytes]:
        """
        Get raw segmentation image data as uint16 array.
        
        Args:
            miplevel: Mip level to retrieve
            minx, maxx, miny, maxy, minz, maxz: Bounding box coordinates
            immediate: Use immediate mode (don't wait for disk cache)
            request_load: Request loading from disk if not in cache
            
        Returns:
            Raw bytes (uint16 little-endian) or None on error
        """
        cmd = GETSEGIMAGERAWIMMEDIATE if immediate else GETSEGIMAGERAW
        
        if immediate:
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz) + self._encode_uint32(int(request_load))
        else:
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(cmd, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        self.last_error = 0
        return data

    def get_seg_image_rle(self, miplevel: int, minx: int, maxx: int,
                          miny: int, maxy: int, minz: int, maxz: int,
                          surf_only: bool = False, immediate: bool = False,
                          request_load: bool = False) -> Optional[bytes]:
        """
        Get RLE-encoded segmentation image data.
        
        Returns:
            RLE-encoded bytes (uint16 pairs: value, count) or None on error
        """
        if immediate:
            cmd = GETSEGIMAGERLEIMMEDIATE
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz) + self._encode_uint32(int(request_load))
        else:
            cmd = GETSEGIMAGESURFRLE if surf_only else GETSEGIMAGERLE
            payload = self._encode_uint32(miplevel) + self._encode_uint32(minx) + \
                     self._encode_uint32(maxx) + self._encode_uint32(miny) + \
                     self._encode_uint32(maxy) + self._encode_uint32(minz) + \
                     self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(cmd, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        self.last_error = 0
        return data

    def get_seg_image_rle_decoded(self, miplevel: int, minx: int, maxx: int,
                                   miny: int, maxy: int, minz: int, maxz: int,
                                   surf_only: bool = False, immediate: bool = False,
                                   request_load: bool = False):
        """
        Get segmentation image with RLE decoding applied.
        
        Returns:
            numpy array of shape (maxy-miny+1, maxx-minx+1, maxz-minz+1) dtype uint16
        """
        
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                          surf_only, immediate, request_load)
        
        if rle_data is None:
            return None
        
        # Decode RLE
        rle = np.frombuffer(rle_data, dtype=np.uint16)
        
        # Calculate expected size
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        expected_size = size_x * size_y * size_z
        
        result = np.zeros(expected_size, dtype=np.uint16)
        dp = 0
        
        for sp in range(0, len(rle), 2):
            val = rle[sp]
            count = rle[sp + 1]
            result[dp:dp + count] = val
            dp += count
        
        # Reshape to (X, Y, Z) then transpose to (Y, X, Z) to match MATLAB
        result = result.reshape(size_x, size_y, size_z)
        result = np.transpose(result, (1, 0, 2))
        
        return result

    def get_rle_count_unique(self, miplevel: int, minx: int, maxx: int,
                         miny: int, maxy: int, minz: int, maxz: int,
                         surf_only: bool = False, immediate: bool = False,
                         request_load: bool = False) -> tuple:
        """
        Get unique segment IDs and their voxel counts from RLE data without full decoding.
        
        Args:
            miplevel: Mip level
            minx, maxx, miny, maxy, minz, maxz: Bounding box
            surf_only: Only get surface voxels
            immediate: Use immediate mode
            request_load: Request loading from disk
        
        Returns:
            Tuple of (values, counts) where:
            - values: List of segment IDs present
            - counts: List of voxel counts for each ID
            Returns ([], []) on failure
        """
        
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                        surf_only, immediate, request_load)
        
        if rle_data is None:
            return [], []
        
        rle = np.frombuffer(rle_data, dtype=np.uint16)
        
        if len(rle) == 0:
            return [], []
        
        # Get max segment value to size accumulator
        max_seg_val = np.max(rle[0::2])
        counts_array = np.zeros(max_seg_val + 1, dtype=np.uint64)
        
        # Accumulate counts
        for i in range(0, len(rle), 2):
            val = rle[i]
            count = rle[i + 1]
            counts_array[val] += count
        
        # Extract non-zero entries
        nonzero_indices = np.where(counts_array > 0)[0]
        values = nonzero_indices.tolist()
        counts = counts_array[nonzero_indices].tolist()
        
        return values, counts
    
    def get_seg_image_rle_decoded_count_unique(self, miplevel: int, minx: int, maxx: int,
                                           miny: int, maxy: int, minz: int, maxz: int,
                                           surf_only: bool = False, immediate: bool = False,
                                           request_load: bool = False) -> tuple:
        """
        Get decoded RLE image along with unique segment counts.
        
        Returns:
            Tuple of (seg_image, values, counts) where:
            - seg_image: numpy array of shape (Y, X, Z)
            - values: List of unique segment IDs
            - counts: List of voxel counts
            Returns (None, [], []) on failure
        """
        
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                        surf_only, immediate, request_load)
        
        if rle_data is None:
            return None, [], []
        
        rle = np.frombuffer(rle_data, dtype=np.uint16)
        
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        expected_size = size_x * size_y * size_z
        
        # Get max value for count array
        max_seg_val = np.max(rle[0::2])
        counts_array = np.zeros(max_seg_val + 1, dtype=np.uint64)
        
        # Decode RLE and count simultaneously
        seg_image = np.zeros(expected_size, dtype=np.uint16)
        dp = 0
        
        for sp in range(0, len(rle), 2):
            val = rle[sp]
            count = rle[sp + 1]
            seg_image[dp:dp + count] = val
            counts_array[val] += count
            dp += count
        
        # Reshape
        seg_image = seg_image.reshape(size_x, size_y, size_z)
        seg_image = np.transpose(seg_image, (1, 0, 2))
        
        # Extract unique values and counts
        nonzero_indices = np.where(counts_array > 0)[0]
        values = nonzero_indices.tolist()
        counts = counts_array[nonzero_indices].tolist()
        
        return seg_image, values, counts
    
    def get_seg_image_rle_decoded_bboxes(self, miplevel: int, minx: int, maxx: int,
                                     miny: int, maxy: int, minz: int, maxz: int,
                                     surf_only: bool = False, immediate: bool = False,
                                     request_load: bool = False) -> tuple:
        """
        Get decoded RLE image with bounding boxes for each segment.

        Returns:
            Tuple of (seg_image, values, counts, bboxes) where:
            - seg_image: numpy array of shape (X, Y, Z) matching VAST coordinate system
            - values: List of unique segment IDs
            - counts: List of voxel counts
            - bboxes: List of bounding boxes [xmin, ymin, zmin, xmax, ymax, zmax]
                     Coordinates are 0-indexed and match seg_image dimensions
            Returns (None, [], [], []) on failure
        """
        
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                        surf_only, immediate, request_load)
        
        if rle_data is None:
            return None, [], [], []
        
        rle = np.frombuffer(rle_data, dtype=np.uint16)
        
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        expected_size = size_x * size_y * size_z
        
        # Get max value for arrays
        max_seg_val = np.max(rle[0::2])
        counts_array = np.zeros(max_seg_val + 1, dtype=np.uint64)
        bboxes_array = np.full((max_seg_val + 1, 6), -1, dtype=np.int32)
        
        # Decode RLE and compute bboxes simultaneously
        seg_image = np.zeros(expected_size, dtype=np.uint16)
        dp = 0
        
        for sp in range(0, len(rle), 2):
            val = rle[sp]
            count = int(rle[sp + 1])
            
            # Write to image
            seg_image[dp:dp + count] = val
            counts_array[val] += count
            
            # Compute bounding box for this run
            start_idx = dp
            end_idx = dp + count - 1
            
            # Convert linear indices to 3D coordinates
            z1 = start_idx // (size_x * size_y)
            r = start_idx % (size_x * size_y)
            y1 = r // size_x
            x1 = r % size_x
            
            z2 = end_idx // (size_x * size_y)
            r = end_idx % (size_x * size_y)
            y2 = r // size_x
            x2 = r % size_x
            
            # Determine actual bbox for this run
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            zmin = min(z1, z2)
            zmax = max(z1, z2)
            
            # Handle runs that span multiple planes
            if zmax > zmin:
                xmin, xmax = 0, size_x - 1
                ymin, ymax = 0, size_y - 1
            elif ymax > ymin:
                xmin, xmax = 0, size_x - 1
            
            # Update segment bbox
            if bboxes_array[val, 0] == -1:
                # First occurrence
                bboxes_array[val] = [xmin, ymin, zmin, xmax, ymax, zmax]
            else:
                # Expand existing bbox
                bboxes_array[val, 0] = min(bboxes_array[val, 0], xmin)
                bboxes_array[val, 1] = min(bboxes_array[val, 1], ymin)
                bboxes_array[val, 2] = min(bboxes_array[val, 2], zmin)
                bboxes_array[val, 3] = max(bboxes_array[val, 3], xmax)
                bboxes_array[val, 4] = max(bboxes_array[val, 4], ymax)
                bboxes_array[val, 5] = max(bboxes_array[val, 5], zmax)
            
            dp += count
        
        # Reshape image to (X, Y, Z) - matches bbox coordinate system
        seg_image = seg_image.reshape(size_x, size_y, size_z)
        # Note: Bboxes are in [xmin, ymin, zmin, xmax, ymax, zmax] format
        # This ordering is consistent with the reshaped image dimensions
        
        # Extract results for non-zero segments
        nonzero_indices = np.where(counts_array > 0)[0]
        values = nonzero_indices.tolist()
        counts = counts_array[nonzero_indices].tolist()
        bboxes = bboxes_array[nonzero_indices].tolist()
        
        return seg_image, values, counts, bboxes

    def set_seg_translation(self, source_array: List[int], target_array: List[int]) -> bool:
        """
        Set segmentation translation table for image retrieval.
        
        Maps source segment IDs to target IDs during image transfer.
        Pass empty arrays to clear translation.
        """
        if len(source_array) != len(target_array):
            self.last_error = 50
            return False
        
        # Interleave source and target arrays
        translate = []
        for src, tgt in zip(source_array, target_array):
            translate.append(src)
            translate.append(tgt)
        
        payload = b""
        for val in translate:
            payload += self._encode_uint32(val)
        
        msg_type, data = self.send_command(SETSEGTRANSLATION, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_em_image_raw(self, layer_nr: int, miplevel: int, minx: int, maxx: int,
                     miny: int, maxy: int, minz: int, maxz: int,
                     immediate: bool = False, request_load: bool = False) -> Optional[bytes]:
        """
        Get raw EM/image layer data.
        
        Args:
            layer_nr: Layer number
            miplevel: Mip level
            minx, maxx, miny, maxy, minz, maxz: Bounding box
            immediate: Use immediate mode (don't wait for disk cache)
            request_load: Request loading from disk if not in cache
        
        Returns:
            Raw bytes (uint8), or None on failure
        """
        cmd = GETEMIMAGERAWIMMEDIATE if immediate else GETEMIMAGERAW
        
        if immediate:
            payload = self._encode_uint32(layer_nr) + \
                    self._encode_uint32(miplevel) + \
                    self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                    self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                    self._encode_uint32(minz) + self._encode_uint32(maxz) + \
                    self._encode_uint32(int(request_load))
        else:
            payload = self._encode_uint32(layer_nr) + \
                    self._encode_uint32(miplevel) + \
                    self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                    self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                    self._encode_uint32(minz) + self._encode_uint32(maxz)
        
        msg_type, data = self.send_command(cmd, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        self.last_error = 0
        return data

    def get_em_image(self, layer_nr: int, miplevel: int, minx: int, maxx: int,
                    miny: int, maxy: int, minz: int, maxz: int,
                    immediate: bool = False, request_load: bool = False) -> Optional[np.ndarray]:
        """
        Get EM/image layer data as numpy array with proper reshaping.
        
        Returns:
            Numpy array with shape:
            - 1 byte/pixel: (maxy-miny+1, maxx-minx+1, [maxz-minz+1])
            - 3 bytes/pixel: (maxy-miny+1, maxx-minx+1, [maxz-minz+1], 3) RGB
            - 4 bytes/pixel: (maxy-miny+1, maxx-minx+1, [maxz-minz+1]) uint32
            - 8 bytes/pixel: (maxy-miny+1, maxx-minx+1, [maxz-minz+1]) uint64
            Or None on failure
        """
        
        raw_data = self.get_em_image_raw(layer_nr, miplevel, minx, maxx, miny, maxy, 
                                        minz, maxz, immediate, request_load)
        
        if raw_data is None:
            return None
        
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        
        # Determine bytes per pixel
        total_voxels = size_x * size_y * size_z
        bytes_per_pixel = len(raw_data) // total_voxels
        
        if bytes_per_pixel == 1:
            # Grayscale 8-bit
            arr = np.frombuffer(raw_data, dtype=np.uint8)
            if minz == maxz:
                # 2D image
                result = arr.reshape(size_x, size_y)
                result = np.transpose(result, (1, 0))
            else:
                # 3D volume
                result = arr.reshape(size_x, size_y, size_z)
                result = np.transpose(result, (1, 0, 2))
        
        elif bytes_per_pixel == 3:
            # RGB 24-bit
            arr = np.frombuffer(raw_data, dtype=np.uint8)
            if minz == maxz:
                # 2D image
                result = arr.reshape(3, size_x, size_y)
                result = np.transpose(result, (2, 1, 0))
                result = np.flip(result, axis=2)  # RGB order
            else:
                # 3D volume
                result = arr.reshape(3, size_x, size_y, size_z)
                result = np.transpose(result, (2, 1, 3, 0))
                result = np.flip(result, axis=3)  # RGB order
        
        elif bytes_per_pixel == 4:
            # 32-bit
            arr = np.frombuffer(raw_data, dtype=np.uint32)
            if minz == maxz:
                result = arr.reshape(size_x, size_y)
                result = np.transpose(result, (1, 0))
            else:
                result = arr.reshape(size_x, size_y, size_z)
                result = np.transpose(result, (1, 0, 2))
        
        elif bytes_per_pixel == 8:
            # 64-bit
            arr = np.frombuffer(raw_data, dtype=np.uint64)
            if minz == maxz:
                result = arr.reshape(size_x, size_y)
                result = np.transpose(result, (1, 0))
            else:
                result = arr.reshape(size_x, size_y, size_z)
                result = np.transpose(result, (1, 0, 2))
        
        else:
            self.last_error = 2
            return None
        
        return result

    def get_screenshot_image_raw(self, miplevel: int, minx: int, maxx: int,
                                miny: int, maxy: int, minz: int, maxz: int,
                                collapse_seg: bool = False) -> Optional[bytes]:
        """
        Get raw screenshot image from VAST display.
        
        Args:
            miplevel: Mip level
            minx, maxx, miny, maxy, minz, maxz: Region coordinates
            collapse_seg: If True, collapse segmentation to single color per segment
        
        Returns:
            Raw RGB bytes (uint8), or None on failure
        """
        payload = self._encode_uint32(miplevel) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz) + \
                self._encode_uint32(int(collapse_seg))
        
        msg_type, data = self.send_command(GETSCREENSHOTIMAGERAW, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        self.last_error = 0
        return data

    def get_screenshot_image(self, miplevel: int, minx: int, maxx: int,
                            miny: int, maxy: int, minz: int, maxz: int,
                            collapse_seg: bool = False, use_rle: bool = False) -> Optional[np.ndarray]:
        """
        Get screenshot image as numpy array.
        
        Args:
            miplevel: Mip level
            minx, maxx, miny, maxy, minz, maxz: Region coordinates
            collapse_seg: If True, collapse segmentation to single color per segment
            use_rle: Use RLE encoding for transfer
        
        Returns:
            Numpy array of shape (maxy-miny+1, maxx-minx+1, [maxz-minz+1], 3) RGB
            Or None on failure
        """
        
        if use_rle:
            raw_data = self.get_screenshot_image_rle(miplevel, minx, maxx, miny, maxy, 
                                                    minz, maxz, collapse_seg)
        else:
            raw_data = self.get_screenshot_image_raw(miplevel, minx, maxx, miny, maxy, 
                                                    minz, maxz, collapse_seg)
        
        if raw_data is None:
            return None
        
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        
        arr = np.frombuffer(raw_data, dtype=np.uint8)
        
        if minz == maxz:
            # 2D image
            result = arr.reshape(3, size_x, size_y)
            result = np.transpose(result, (2, 1, 0))
            result = np.flip(result, axis=2)  # RGB order
        else:
            # 3D volume
            result = arr.reshape(3, size_x, size_y, size_z)
            result = np.transpose(result, (2, 1, 3, 0))
            result = np.flip(result, axis=3)  # RGB order
        
        return result

    def get_screenshot_image_rle(self, miplevel: int, minx: int, maxx: int,
                                miny: int, maxy: int, minz: int, maxz: int,
                                collapse_seg: bool = False) -> Optional[bytes]:
        """
        Get RLE-encoded screenshot image from VAST display.
        
        Args:
            miplevel: Mip level
            minx, maxx, miny, maxy, minz, maxz: Region coordinates
            collapse_seg: If True, collapse segmentation to single color per segment
        
        Returns:
            RLE-encoded RGB bytes, or None on failure
        """
        payload = self._encode_uint32(miplevel) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz) + \
                self._encode_uint32(int(collapse_seg))
        
        msg_type, data = self.send_command(GETSCREENSHOTIMAGERLE, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        # Check if already uncompressed (RLE was same size or larger)
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        expected_size = size_x * size_y * size_z * 3
        
        if len(data) == expected_size:
            # Already raw
            self.last_error = 0
            return data
        
        # Decode RLE
        rle_data = np.frombuffer(data, dtype=np.uint8)
        
        decoded = np.zeros(expected_size, dtype=np.uint8)
        dp = 0
        
        for sp in range(0, len(rle_data), 4):
            if sp + 3 >= len(rle_data):
                break
            r = rle_data[sp]
            g = rle_data[sp + 1]
            b = rle_data[sp + 2]
            count = rle_data[sp + 3]
            
            for i in range(count):
                if dp + 2 < len(decoded):
                    decoded[dp] = r
                    decoded[dp + 1] = g
                    decoded[dp + 2] = b
                    dp += 3
        
        self.last_error = 0
        return decoded.tobytes()

    def order_screenshot_image(self, miplevel: int, minx: int, maxx: int,
                          miny: int, maxy: int, minz: int, maxz: int,
                          collapse_seg: bool = False) -> bool:
        """
        Order screenshot image (async - returns immediately).
        Must call pickup_screenshot_image() next to receive data.
        Do not call other API functions between order and pickup!
        
        Args:
            miplevel: Mip level
            minx, maxx, miny, maxy, minz, maxz: Region coordinates
            collapse_seg: Collapse segmentation colors
        
        Returns:
            True (command sent successfully)
        """
        payload = self._encode_uint32(miplevel) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz) + \
                self._encode_uint32(int(collapse_seg))
        
        # Send command but don't wait for response
        client = self.client
        if client is None:
            return False
        
        total_len = len(payload) + 4
        header = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", GETSCREENSHOTIMAGERAW)
        message = header + payload
        
        client.sendall(message)
        return True
    
    def pickup_screenshot_image(self, miplevel: int, minx: int, maxx: int,
                           miny: int, maxy: int, minz: int, maxz: int,
                           collapse_seg: bool = False) -> Optional[np.ndarray]:
        """
        Receive screenshot image after calling order_screenshot_image().
        Parameters must match those used in order call.
        
        Returns:
            Numpy array of shape (maxy-miny+1, maxx-minx+1, [maxz-minz+1], 3) RGB
            Or None on failure
        """
        
        # Now receive the response
        client = self.client
        if client is None:
            return None
        
        # Receive response header
        hdr = client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            self.last_error = 2
            return None
        
        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type = struct.unpack("<I", hdr[12:16])[0]
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        expected = total_len - 4
        payload_bytes = bytearray()
        while len(payload_bytes) < expected:
            chunk = client.recv(expected - len(payload_bytes))
            if not chunk:
                self.last_error = 2
                return None
            payload_bytes.extend(chunk)
        
        # Reshape to image
        size_x = maxx - minx + 1
        size_y = maxy - miny + 1
        size_z = maxz - minz + 1
        
        arr = np.frombuffer(bytes(payload_bytes), dtype=np.uint8)
        
        if minz == maxz:
            result = arr.reshape(3, size_x, size_y)
            result = np.transpose(result, (2, 1, 0))
            result = np.flip(result, axis=2)
        else:
            result = arr.reshape(3, size_x, size_y, size_z)
            result = np.transpose(result, (2, 1, 3, 0))
            result = np.flip(result, axis=3)
        
        self.last_error = 0
        return result
    
    def set_seg_image_raw(self, miplevel: int, minx: int, maxx: int,
                      miny: int, maxy: int, minz: int, maxz: int,
                      seg_image: np.ndarray) -> bool:
        """
        Write raw segmentation image data to VAST.
        
        Args:
            miplevel: Mip level to write to
            minx, maxx, miny, maxy, minz, maxz: Bounding box
            seg_image: Numpy array of shape (maxy-miny+1, maxx-minx+1, maxz-minz+1), dtype uint16
        
        Returns:
            True on success, False on failure
        """
        
        # Verify dimensions
        expected_shape = (maxy - miny + 1, maxx - minx + 1, maxz - minz + 1)
        if seg_image.shape != expected_shape:
            self.last_error = 13  # Data size mismatch
            return False
        
        # Transpose from (Y, X, Z) to (X, Y, Z) for transmission
        seg_transposed = np.transpose(seg_image, (1, 0, 2))
        
        # Flatten and convert to bytes
        seg_flat = seg_transposed.flatten().astype(np.uint16)
        seg_bytes = seg_flat.tobytes()
        
        # Build payload
        params = self._encode_uint32(miplevel) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz)
        
        # Add data with type tag (raw binary data)
        payload = params + b'\x05' + len(seg_bytes).to_bytes(4, 'little') + seg_bytes
        
        msg_type, data = self.send_command(SETSEGIMAGERAW, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def set_seg_image_rle(self, miplevel: int, minx: int, maxx: int,
                        miny: int, maxy: int, minz: int, maxz: int,
                        seg_image: np.ndarray) -> bool:
        """
        Write RLE-encoded segmentation image data to VAST.
        
        Args:
            miplevel: Mip level to write to
            minx, maxx, miny, maxy, minz, maxz: Bounding box
            seg_image: Numpy array of shape (maxy-miny+1, maxx-minx+1, maxz-minz+1), dtype uint16
        
        Returns:
            True on success, False on failure
        """
        
        # Verify dimensions
        expected_shape = (maxy - miny + 1, maxx - minx + 1, maxz - minz + 1)
        if seg_image.shape != expected_shape:
            self.last_error = 13  # Data size mismatch
            return False
        
        # Transpose from (Y, X, Z) to (X, Y, Z) for transmission
        seg_transposed = np.transpose(seg_image, (1, 0, 2))
        seg_flat = seg_transposed.flatten().astype(np.uint16)
        
        # RLE encode
        rle_data = []
        if len(seg_flat) > 0:
            current_val = seg_flat[0]
            count = 1
            
            for i in range(1, len(seg_flat)):
                if seg_flat[i] == current_val and count < 65535:
                    count += 1
                else:
                    rle_data.append(current_val)
                    rle_data.append(count)
                    current_val = seg_flat[i]
                    count = 1
            
            # Add last run
            rle_data.append(current_val)
            rle_data.append(count)
        
        rle_array = np.array(rle_data, dtype=np.uint16)
        
        # Check if RLE is actually smaller
        if len(rle_array) >= len(seg_flat):
            # Fall back to raw
            return self.set_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz, seg_image)
        
        # Build payload
        params = self._encode_uint32(miplevel) + \
                self._encode_uint32(minx) + self._encode_uint32(maxx) + \
                self._encode_uint32(miny) + self._encode_uint32(maxy) + \
                self._encode_uint32(minz) + self._encode_uint32(maxz)
        
        rle_bytes = rle_array.tobytes()
        payload = params + b'\x05' + len(rle_bytes).to_bytes(4, 'little') + rle_bytes
        
        msg_type, data = self.send_command(SETSEGIMAGERLE, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    def get_pixel_value(self, layer_nr: int, miplevel: int, 
                    x: int, y: int, z: int) -> int:
        """
        Get pixel value at specific coordinates.
        
        Args:
            layer_nr: Layer number
            miplevel: Mip level
            x, y, z: Coordinates (full resolution)
        
        Returns:
            Pixel value (uint64), or 0 on failure
        """
        payload = self._encode_uint32(layer_nr) + \
                self._encode_uint32(miplevel) + \
                self._encode_uint32(x) + \
                self._encode_uint32(y) + \
                self._encode_uint32(z)
        
        msg_type, data = self.send_command(GETPIXELVALUE, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        
        parsed = self.parse_payload(data)
        uint64s = parsed.get("uint64s", [])
        
        if len(uint64s) == 1:
            self.last_error = 0
            return uint64s[0]
        else:
            self.last_error = 2
            return 0

    ##########################
    # Execute VAST Functions #
    ##########################

    def execute_fill(self, source_layer_nr: int, target_layer_nr: int,
                    x: int, y: int, z: int, mip: int) -> bool:
        """
        Perform masked filling operation.
        
        Args:
            source_layer_nr: Source layer for masking
            target_layer_nr: Target segmentation layer
            x, y, z: Seed coordinates (full resolution/mip 0)
            mip: Mip level at which to execute fill
        
        Returns:
            True on success, False on failure
        
        Note: This is a blocking operation that may take a long time.
        Use filling properties set via set_filling_properties().
        """
        payload = self._encode_uint32(source_layer_nr) + \
                self._encode_uint32(target_layer_nr) + \
                self._encode_uint32(x) + \
                self._encode_uint32(y) + \
                self._encode_uint32(z) + \
                self._encode_uint32(mip)
        
        msg_type, data = self.send_command(EXECUTEFILL, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success


    def execute_limited_fill(self, source_layer_nr: int, target_layer_nr: int,
                            x: int, y: int, z: int, mip: int) -> bool:
        """
        Perform limited masked filling operation.
        
        Args:
            source_layer_nr: Source layer for masking
            target_layer_nr: Target segmentation layer
            x, y, z: Seed coordinates (full resolution/mip 0)
            mip: Mip level at which to execute fill
        
        Returns:
            True on success, False on failure
        
        Note: Limited fill stops at certain boundaries.
        This is a blocking operation that may take a long time.
        """
        payload = self._encode_uint32(source_layer_nr) + \
                self._encode_uint32(target_layer_nr) + \
                self._encode_uint32(x) + \
                self._encode_uint32(y) + \
                self._encode_uint32(z) + \
                self._encode_uint32(mip)
        
        msg_type, data = self.send_command(EXECUTELIMITEDFILL, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success


    def execute_canvas_paint_stroke(self, coords: np.ndarray) -> bool:
        """
        Perform a paint stroke on the selected segmentation layer.
        
        Args:
            coords: Numpy array of shape (n, 2) with window coordinates [x, y]
                    Each row is one point in the paint stroke.
        
        Returns:
            True on success, False on failure
        
        Note: Uses window coordinates. Region must be loaded in 2D canvas.
        """
        
        if coords.ndim != 2 or coords.shape[1] != 2:
            self.last_error = 14  # Parameter out of bounds
            return False
        
        # Flatten to row-major order and convert to uint32
        coords_flat = coords.T.flatten().astype(np.uint32)
        coords_bytes = coords_flat.tobytes()
        
        # Encode dimensions and data
        payload = self._encode_uint32(coords.shape[0]) + \
                self._encode_uint32(coords.shape[1]) + \
                b'\x05' + len(coords_bytes).to_bytes(4, 'little') + coords_bytes
        
        msg_type, data = self.send_command(EXECUTECANVASPAINTSTROKE, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success


    def execute_start_auto_skeletonization(self, tool_layer_nr: int, mip: int,
                                        node_distance_mu: float, node_step: int,
                                        region_padding_mu: float) -> bool:
        """
        Start automatic skeletonization.
        
        Args:
            tool_layer_nr: Tool layer number to use
            mip: Mip level for processing
            node_distance_mu: Distance between nodes in micrometers
            node_step: Step size for node placement
            region_padding_mu: Padding around region in micrometers
        
        Returns:
            True on success, False on failure
        
        Note: Starts from selected node in annotation layer.
            Requires selected node to have a parent node.
        """
        payload = self._encode_uint32(tool_layer_nr) + \
                self._encode_uint32(mip) + \
                self._encode_uint32(node_step) + \
                self._encode_double(node_distance_mu) + \
                self._encode_double(region_padding_mu)
        
        msg_type, data = self.send_command(EXECUTESTARTAUTOSKELETONIZATION, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success


    def execute_stop_auto_skeletonization(self) -> bool:
        """
        Stop the currently running auto-skeletonization.
        
        Returns:
            True on success, False if not running (error code 24)
        """
        msg_type, data = self.send_command(EXECUTESTOPAUTOSKELETONIZATION)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success


    def execute_get_auto_skeletonization_state(self) -> dict:
        """
        Get the current state of auto-skeletonization.
        
        Returns:
            Dict with keys:
            - isdone: 1 if completed, 0 if running
            - iswaiting: 1 if waiting for data load, 0 otherwise
            - nrnodesadded: Number of nodes added so far
            Returns empty dict on failure
        """
        msg_type, data = self.send_command(EXECUTEISAUTOSKELETONIZATIONDONE)
        
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        
        parsed = self.parse_payload(data)
        uints = parsed.get("uints", [])
        
        if len(uints) == 3:
            self.last_error = 0
            return {
                "isdone": uints[0],
                "iswaiting": uints[1],
                "nrnodesadded": uints[2],
            }
        else:
            self.last_error = 2
            return {}
