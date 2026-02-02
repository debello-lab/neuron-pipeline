import socket
import struct
import numpy as np
from typing import List, Tuple, Any, Optional, Dict, Union

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
        self.is_connected      = 0;
        self.last_error        = 0;
        self.this_versionnr    = 5; #Version 5 is for VAST Lite 1.3, 1.4, 1.5
        self.this_subversionnr = 14;
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
            self.is_connected = 1
        except Exception as e:
            self.is_connected = 0
            print(f"Connection error: {e}")
        return self.is_connected

    def disconnect(self):
        if self.client_socket is not None:
            try:
                self.client_socket.close()
                self.is_connected = 0
            except Exception as e:
                print(f"Disconnection error: {e}")
        
        return 1

    def get_last_error(self):
        return self.last_error

    def set_error_popups_enabled(self, code, enabled: bool) -> bool:
        """
        Enable or disable error popups in VAST for a specific error code.
        Corresponds to MATLAB seterrorpopupenabled(code, enabled).
        """
        payload = self.bytes_from_uint32(code) + self.bytes_from_uint32(1 if enabled else 0)
        msg_type, data = self.send_command(self.SETERRORPOPUPSENABLED, payload)

        if msg_type == 1:
            self.last_error = 0
            return True
        else:
            self.last_error = msg_type
            print(f"Error setting error popup enabled: {ERROR_CODES.get(msg_type, 'Unknown error')}")
            return False
        
    ######################################################################
    # GETINFO

    def get_info(self) -> Optional[Dict[str, Any]]:
        msg_type, data = self.send_command(self.GETINFO)
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        parsed = self.parse_payload(data)

        if(len(parsed["uints"])   == 7 and
           len(parsed["doubles"]) == 3 and
           len(parsed["ints"])    == 3):
            self.last_error = 0
            uintdata, doubledata, intdata = parsed["uints"], parsed["doubles"], parsed["ints"]
            return {
                "datasizex": uintdata[0],
                "datasizey": uintdata[1],
                "datasizez": uintdata[2],
                "voxelsizex": doubledata[0],
                "voxelsizey": doubledata[1],
                "voxelsizez": doubledata[2],
                "cubesizex": uintdata[3],
                "cubesizey": uintdata[4],
                "cubesizez": uintdata[5],
                "currentviewx": intdata[0],
                "currentviewy": intdata[1],
                "currentviewz": intdata[2],
                "nrofmiplevels": uintdata[6],
            }
        else:
            self.last_error = 2
            return None

    def get_hardware_info(self) -> dict:
        msg_type, data = self.send_command(GETHARDWAREINFO)
        if msg_type == 21:
            self.last_error = 21
            print(f"Error getting hardware info: {errorCodes[msg_type]}")
            return {}
        try:
            parsed = self.parse_payload(data)

            u32    = parsed["uints"]
            f64    = parsed["doubles"]
            i32    = parsed["ints"]
            texts  = parsed["text"]

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

    ######################################################################
    # 2: getnumberofsegments()
    def get_number_of_segments(self) -> Optional[int]:
        msg_type, data = self.send_command(self.GETNUMBEROFSEGMENTS)
    
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        parsed = self.parse_payload(data)
        
        if len(parsed["uints"]) == 1:
            self.last_error = 0
            return parsed["uints"][0]
        else:
            self.last_error = 2  # unexpected data
            return None

    ######################################################################
    # 3: getsegmentdata(segment_id)
    def get_segment_data(self, segment_id: int) -> Optional[Dict]:
        payload = self.bytes_from_uint32(segment_id)
        msg_type, data = self.send_command(self.GETSEGMENTDATA, payload)
    
        if msg_type != 1:
            self.last_error = msg_type
            return None
    
        parsed = self.parse_payload(data)
    
        if len(parsed["ints"]) == 9 and len(parsed["uints"]) == 9:
            self.last_error = 0
            return {
            "id": parsed["uints"][0],
            "flags": parsed["uints"][1],
            "col1": parsed["uints"][2],
            "col2": parsed["uints"][3],
            "anchorpoint": parsed["ints"][0:3],      # indices 0, 1, 2
            "hierarchy": parsed["uints"][4:8],       # indices 4, 5, 6, 7
            "collapsednr": parsed["uints"][8],
            "boundingbox": parsed["ints"][3:9],      # indices 3, 4, 5, 6, 7, 8
            }
        else:
            self.last_error = 2  # unexpected data
            return None

    ######################################################################
    # 4: getsegmentname(segment_id)
    def get_segment_name(self, segment_id: int) -> Optional[str]:
        payload = self.bytes_from_uint32(segment_id)
        msg_type, data = self.send_command(self.GETSEGMENTNAME, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        parsed = self.parse_payload(data)
        
        if len(parsed["text"]) == 1:
            self.last_error = 0
            return parsed["text"][0]
        else:
            self.last_error = 2  # unexpected data
            return None

    ######################################################################
    # GETNROFLAYERS = 12
    def get_nrof_layers(self) -> int:
        msg_type, data = self.send_command(self.GETNROFLAYERS)

        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        parsed = self.parse_payload(data)

        if len(parsed["uints"]) == 1:
            self.last_error = 0
            return parsed["uints"][0]
        else:
            self.last_error = 2
            return 0
    
    ######################################################################
    # GETLAYERINFO = 13
    def get_layer_info(self, layer_nr: int) -> Optional[Dict[str, any]]:
        payload = self.bytes_from_uint32(layer_nr)
        msg_type, data = self.send_command(self.GETLAYERINFO, payload)

        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        parsed = self.parse_payload(data)

        if (len(parsed["ints"]) == 8 and 
                len(parsed["uints"]) == 7 and 
                len(parsed["doubles"]) == 3):
                
                self.last_error = 0
                i, u, d = parsed["ints"], parsed["uints"], parsed["doubles"]
                return {
                    "type": i[0],
                    "editable": i[1],
                    "visible": i[2],
                    "brightness": i[3],
                    "contrast": i[4],
                    "opacitylevel": d[0],
                    "brightnesslevel": d[1],
                    "contrastlevel": d[2],
                    "blendmode": i[5],
                    "blendoradd": i[6],
                    "tintcolor": u[0],
                    "name": parsed["last_text"],
                    "redtargetcolor": u[1],
                    "greentargetcolor": u[2],
                    "bluetargetcolor": u[3],
                    "bytesperpixel": u[4],
                    "ischanged": i[7],
                    "inverted": u[5],
                    "solomode": u[6],
                }
        else:
            self.last_error = 2  # unexpected data
            return None

    ######################################################################
    # GETALLSEGMENTDATA = 14
    def get_all_segment_data(self) -> Optional[List[Dict[str, any]]]:
        msg_type, data = self.send_command(self.GETALLSEGMENTDATA)
    
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        # Parse the raw binary data directly (no typed payload parsing)
        # Data format: [num_segments (uint32)][segment_0_data][segment_1_data]...
        # Each segment is 17 uint32/int32 values (68 bytes)
        
        if len(data) < 4:
            self.last_error = 2
            return None
        
        # First uint32 is the number of segments
        num_segments = struct.unpack("<I", data[0:4])[0]
        
        expected_len = 4 + (num_segments * 68)  # 4 bytes for count + 68 bytes per segment
        if len(data) < expected_len:
            self.last_error = 2
            return None
        
        segdata = []
        offset = 4  # Start after the segment count
        
        for i in range(num_segments):
            # Read 17 values (mix of uint32 and int32)
            # Extract as raw bytes first
            seg_bytes = data[offset:offset+68]
            
            # Unpack as uint32 for unsigned fields
            u = struct.unpack("<17I", seg_bytes)  # all as uint32
            # Unpack as int32 for signed fields (anchorpoint, boundingbox)
            s = struct.unpack("<17i", seg_bytes)  # all as int32
            
            segdata.append({
                "id": i,  # segment index (0-based)
                "flags": u[0],
                "col1": u[1],
                "col2": u[2],
                "anchorpoint": [s[3], s[4], s[5]],  # x, y, z (signed)
                "hierarchy": [u[6], u[7], u[8], u[9]],  # parent, child, prev, next
                "collapsednr": u[10],
                "boundingbox": [s[11], s[12], s[13], s[14], s[15], s[16]],  # x1,y1,z1,x2,y2,z2 (signed)
            })
            
            offset += 68
        
        self.last_error = 0
        return segdata

    def get_all_segment_data_matrix(self) -> Optional[np.ndarray]:
        msg_type, data = self.send_command(self.GETALLSEGMENTDATA)

        if msg_type != 1:
            self.last_error = msg_type
            return None

        if len(data) < 4:
            self.last_error = 2
            return None

        num_segments = struct.unpack("<I", data[0:4])[0]

        expected_len = 4 + (num_segments * 68)
        if len(data) < expected_len:
            self.last_error = 2
            return None

        # Pre allocate the matrix
        segdatamatrix = np.zeros((num_segments, 24), dtype=np.int32)
        offset = 4

        for i in range(num_segments):
            seg_bytes = data[offset:offset+68]
            u = struct.unpack("<17I", seg_bytes)  # uint32
            s = struct.unpack("<17i", seg_bytes)  # int32
            
            segdatamatrix[i, 0] = i  # segment id
            segdatamatrix[i, 1] = u[0]  # flags
            
            # Extract color1 bytes from uint32 (col1)
            segdatamatrix[i, 2] = (u[1] >> 24) & 0xFF  # Red
            segdatamatrix[i, 3] = (u[1] >> 16) & 0xFF  # Green
            segdatamatrix[i, 4] = (u[1] >> 8) & 0xFF   # Blue
            segdatamatrix[i, 5] = u[1] & 0xFF          # Pattern
            
            # Extract color2 bytes from uint32 (col2)
            segdatamatrix[i, 6] = (u[2] >> 24) & 0xFF  # Red
            segdatamatrix[i, 7] = (u[2] >> 16) & 0xFF  # Green
            segdatamatrix[i, 8] = (u[2] >> 8) & 0xFF   # Blue
            segdatamatrix[i, 9] = u[2] & 0xFF          # Pattern
            
            # Anchorpoint (signed)
            segdatamatrix[i, 10:13] = [s[3], s[4], s[5]]
            
            # Hierarchy (unsigned)
            segdatamatrix[i, 13:17] = [u[6], u[7], u[8], u[9]]
            
            # Collapsed number
            segdatamatrix[i, 17] = u[10]
            
            # Bounding box (signed)
            segdatamatrix[i, 18:24] = [s[11], s[12], s[13], s[14], s[15], s[16]]
            
            offset += 68
        
        # Skip first segment (index 0), matching MATLAB: segdatamatrix(2:end,:)
        self.last_error = 0
        return segdatamatrix[1:, :]

    ######################################################################
    # GETALLSEGMENTNAMES = 15
    def get_all_segment_names(self) -> Optional[List[str]]:
        msg_type, data = self.send_command(self.GETALLSEGMENTNAMES)
    
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        if len(data) < 4:
            self.last_error = 2
            return None
        
        # First uint32 is the number of names
        num_names = struct.unpack("<I", data[0:4])[0]
        
        segnames = []
        offset = 4  # Start after the count
        
        for i in range(num_names):
            # Find null terminator
            null_pos = offset
            while null_pos < len(data) and data[null_pos] != 0:
                null_pos += 1
            
            if null_pos >= len(data):
                # Incomplete data
                self.last_error = 2
                return None
            
            # Extract string bytes
            name_bytes = data[offset:null_pos]
            try:
                name = name_bytes.decode('utf-8', errors='replace')
            except Exception:
                name = ''.join(chr(b) for b in name_bytes)
            
            segnames.append(name)
            offset = null_pos + 1  # Move past null terminator
        
        self.last_error = 0
        return segnames

    def set_selected_segment_nr(self, segment_id: int) -> bool:
        """
        Select a segment.
        
        Args:
            segment_id: Segment ID to select
        
        Returns:
            True on success, False on failure
        """
        payload = self.bytes_from_uint32(segment_id)
        msg_type, data = self.send_command(self.SETSELECTEDSEGMENTNR, payload)
        
        success = (msg_type == 1)
        self.last_error = 0 if success else msg_type
        return success

    ######################################################################
    # GETSELECTEDSEGMENTNR = 17;
    def get_selected_segment_nr(self) -> int:
        """
        Get the currently selected segment number.
        
        Returns:
            Segment ID, or -1 on failure
        """
        msg_type, data = self.send_command(self.GETSELECTEDSEGMENTNR)
        
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        
        parsed = self.parse_payload(data)
        uints = parsed["uints"]
        
        if len(uints) == 1:
            self.last_error = 0
            return uints[0]
        else:
            self.last_error = 2
            return -1

    ######################################################################
    # GETSELECTEDLAYERNR = 19
    def get_selected_layernr(self) -> Optional[Tuple[int, int, int]]:
        """
        Get the currently selected layer number.
        
        Returns:
            Tuple of (Selected Layer nr, Selected EM layer nr, Selected segment layer nr)

        """
        msg_type, data = self.send_command(self.GETSELECTEDLAYERNR)

        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        parsed = self.parse_payload(data)
        ints = parsed["ints"]
        
        if len(ints) == 3:
          selected_layer = ints[0]
          selected_em_layer = ints[1]
          selected_segment_layer = ints[2]
        elif len(ints) == 5:
            selected_layer = ints[0]
            selected_em_layer = ints[1]
            selected_segment_layer = ints[4]
        else:
            self.last_error = 2
            return None

        self.last_error = 0
        return selected_layer, selected_em_layer, selected_segment_layer

    ######################################################################
    # GETSEGIMAGERAW = 20
    def get_seg_image_raw(  self,
                            miplevel: int,
                            minx: int,
                            maxx: int,
                            miny: int,
                            maxy: int,
                            minz: int,
                            maxz: int,
                            flipflag: int = 0,
                            immediateflag: int = 0,
                            requestloadflag: int = 0
        ) -> Optional[np.ndarray]:
        """
        Get raw segmentation image data for a specified region.
        
        Args:
            miplevel: MIP level to retrieve.
            minx, maxx: X range (inclusive).
            miny, maxy: Y range (inclusive).
            minz, maxz: Z range (inclusive).
            flipflag: If 1, reshape and permute dimensions (default 0).
            immediateflag: If 1, use immediate mode (default 0).
            requestloadflag: Load request flag for immediate mode (default 0).
        
        Returns:
            1D NumPy array of uint16 values, or reshaped array if flipflag=1.
            Returns None on failure.
        """
        if immediateflag == 0:
            payload = self.bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])
            msg_type, data = self.send_command(self.GETSEGIMAGERAW, payload)
        else:
            payload = self.bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz, requestloadflag])
            msg_type, data = self.send_command(self.GETSEGIMAGERAWIMMEDIATE, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        # Convert raw bytes to uint16 array
        if len(data) % 2 != 0:
            self.last_error = 2
            return None
        
        # Unpack as little-endian uint16
        num_values = len(data) // 2
        segimage = np.frombuffer(data, dtype='<u2', count=num_values)
        
        if flipflag == 1:
            # Reshape and permute dimensions
            width = maxx - minx + 1
            height = maxy - miny + 1
            depth = maxz - minz + 1
            
            # Reshape to [width, height, depth]
            segimage = segimage.reshape((width, height, depth), order='F')  
            # Fortran order for MATLAB compatibility
            
            # Permute: swap first two dimensions [height, width, depth]
            segimage = np.transpose(segimage, (1, 0, 2))
            
            # Flatten back to 1D
            segimage = segimage.flatten(order='F')
        
        self.last_error = 0
        return segimage

    ######################################################################
    # GETSEGIMAGERLE = 21; GETSEGIMAGESURFRLE = 22; GETSEGIMAGERLEIMMEDIATE = 25;
    def get_seg_image_rle(
        self,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
        surfonlyflag: int = 0,
        immediateflag: int = 0,
        requestloadflag: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get run-length encoded (RLE) segmentation image data for a specified region.
        
        Args:
            miplevel: MIP level to retrieve.
            minx, maxx: X range (inclusive).
            miny, maxy: Y range (inclusive).
            minz, maxz: Z range (inclusive).
            surfonlyflag: If 1, get surface-only RLE (default 0).
            immediateflag: If 1, use immediate mode (default 0).
            requestloadflag: Load request flag for immediate mode (default 0).
        
        Returns:
            1D NumPy array of uint16 RLE-encoded values, or None on failure.
        """
        payload = self.bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz])
        
        if immediateflag == 0:
            if surfonlyflag == 0:
                msg_type, data = self.send_command(self.GETSEGIMAGERLE, payload)
            else:
                msg_type, data = self.send_command(self.GETSEGIMAGESURFRLE, payload)
        else:
            payload_imm = self.bytes_from_uint32([miplevel, minx, maxx, miny, maxy, minz, maxz, requestloadflag])
            msg_type, data = self.send_command(self.GETSEGIMAGERLEIMMEDIATE, payload_imm)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        # Convert raw bytes to uint16 array
        if len(data) % 2 != 0:
            self.last_error = 2
            return None
        
        # Unpack as little-endian uint16
        num_values = len(data) // 2
        segimage_rle = np.frombuffer(data, dtype='<u2', count=num_values)
        
        self.last_error = 0
        return segimage_rle

    def get_seg_image_rle_decoded(
        self,
        miplevel: int,
        minx: int,
        maxx: int,
        miny: int,
        maxy: int,
        minz: int,
        maxz: int,
        surfonlyflag: int = 0,
        flipflag: int = 0,
        immediateflag: int = 0,
        requestloadflag: int = 0
    ) -> Optional[np.ndarray]:
        """
        Get RLE segmentation image data and decode it to a 3D array.
        
        Args:
            miplevel: MIP level to retrieve.
            minx, maxx: X range (inclusive).
            miny, maxy: Y range (inclusive).
            minz, maxz: Z range (inclusive).
            surfonlyflag: If 1, get surface-only RLE (default 0).
            flipflag: If 1, permute dimensions [height, width, depth] (default 0).
            immediateflag: If 1, use immediate mode (default 0).
            requestloadflag: Load request flag for immediate mode (default 0).
        
        Returns:
            3D NumPy array of uint16 values with shape (width, height, depth),
            or (height, width, depth) if flipflag=1. Returns None on failure.
        """
        segimage_rle = self.get_seg_image_rle(
            miplevel, minx, maxx, miny, maxy, minz, maxz,
            surfonlyflag, immediateflag, requestloadflag
        )
        
        if segimage_rle is None:
            return None
        
        # Decode RLE
        width = maxx - minx + 1
        height = maxy - miny + 1
        depth = maxz - minz + 1
        total_size = width * height * depth
        
        segimage = np.zeros(total_size, dtype=np.uint16)
        dp = 0  # destination pointer (0-based in Python)
        
        # RLE format: [value, count, value, count, ...]
        for sp in range(0, len(segimage_rle), 2):
            if sp + 1 >= len(segimage_rle):
                break
            
            val = segimage_rle[sp]
            num = segimage_rle[sp + 1]
            
            segimage[dp:dp + num] = val
            dp += num
        
        # Reshape to 3D array (Fortran order for MATLAB compatibility)
        segimage = segimage.reshape((width, height, depth), order='F')
        
        if flipflag == 1:
            # Permute: swap first two dimensions [height, width, depth]
            segimage = np.transpose(segimage, (1, 0, 2))
        
        self.last_error = 0
        return segimage


    def set_seg_translation(self, sourcearray: Union[List[int], np.ndarray], targetarray: Union[List[int], np.ndarray]) -> bool:
        """
        Set the segmentation translation for get_seg_image functions.
        
        Before the image is transmitted, all voxels with a value in sourcearray 
        will be translated to the corresponding number in targetarray. 
        Segment numbers which do not appear in sourcearray will be set to 0 (background).
        Call with empty arrays to remove segmentation translation.
        
        Args:
            sourcearray: Array of source segment numbers.
            targetarray: Array of destination segment numbers (must match length of sourcearray).
        
        Returns:
            True on success, False on failure.
        """
        # Convert to numpy arrays for easier handling
        src = np.atleast_1d(np.array(sourcearray, dtype=np.uint32)).flatten()
        tgt = np.atleast_1d(np.array(targetarray, dtype=np.uint32)).flatten()
        
        if len(src) != len(tgt):
            # sourcearray and targetarray must have the same length
            self.last_error = 50
            return False
        
        # Interleave source and target arrays
        translate = np.zeros(2 * len(src), dtype=np.uint32)
        translate[0::2] = src  # Even indices: source values
        translate[1::2] = tgt  # Odd indices: target values
        
        payload = self.bytes_from_data(translate)
        msg_type, data = self.send_command(self.SETSEGTRANSLATION, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return False
        
        parsed = self.parse_payload(data)
        self.last_error = 0
        return True

    ######################################################################
    # GETMIPMAPSCALEFACTORS = 126
    def get_mipmap_scale_factors(self, layer_nr: int) -> Optional[np.ndarray]:
        """
        Get MIP map scale factors for a specific layer.
        
        Args:
            layer_nr: The layer number to query.
        
        Returns:
            NumPy array of shape (num_levels-1, 3) with uint32 scale factors,
            or None on failure. First MIP level (index 0) is skipped.
            Each row contains [scale_x, scale_y, scale_z].
        """
        payload = self.bytes_from_uint32(layer_nr)
        msg_type, data = self.send_command(self.GETMIPMAPSCALEFACTORS, payload)
        
        if msg_type != 1:
            self.last_error = msg_type
            return None
        
        if len(data) < 4:
            self.last_error = 2
            return None
        
        # First uint32 is the number of MIP levels
        num_levels = struct.unpack("<I", data[0:4])[0]
        
        expected_len = 4 + (num_levels * 12)  # 4 bytes for count + 12 bytes (3 uint32) per level
        if len(data) < expected_len:
            self.last_error = 2
            return None
        
        # Pre-allocate matrix
        mipscalematrix = np.zeros((num_levels, 3), dtype=np.uint32)
        
        offset = 4
        
        for i in range(num_levels):
            # Read 3 uint32 values (12 bytes)
            u = struct.unpack("<3I", data[offset:offset+12])
            mipscalematrix[i, :] = u
            offset += 12
        
        # Skip first MIP level (index 0), matching MATLAB: mipscalematrix(1,:)=[]
        self.last_error = 0
        return mipscalematrix[1:, :]








    
    

    def send_command(self, msg_id: int, payload: bytes = b"") -> Tuple[int, bytes]:
        """
        Send a binary command to the VAST API and return the message type and response bytes.
        
        Format sent:
        [0..3]   = b"VAST"
        [4..11]  = uint64 length of data following (payload + 4 bytes for msg_id), little-endian
        [12..15] = uint32 message ID, little-endian
        [16..]   = payload (optional)
        
        Format received:
        [0..3]   = b"VAST"
        [4..11]  = uint64 length of data following, little-endian
        [12..15] = int32 message type/result, little-endian
        [16..]   = payload
        
        Args:
            msg_id: The message/command ID to send.
            payload: Optional payload bytes.
        
        Returns:
            Tuple of (message_type, payload_bytes)
        """
        if self.client_socket is None:
            raise RuntimeError("Not connected to VAST API.")
        
        # Build message
        total_len = len(payload) + 4
        header = b"VAST"
        header += struct.pack("<Q", total_len)  # uint64 little-endian
        header += struct.pack("<I", msg_id)     # uint32 little-endian
        message = header + payload
        
        # Send message
        self.client_socket.sendall(message)
        
        # Receive response header (16 bytes)
        hdr = self.client_socket.recv(16)
        if len(hdr) < 16:
            raise RuntimeError(f"Incomplete header received: {len(hdr)} bytes")
        
        if not hdr.startswith(b"VAST"):
            raise RuntimeError(f"Invalid response header: {hdr[:4]!r}")
        
        # Parse header
        response_len = struct.unpack("<Q", hdr[4:12])[0]   # uint64
        msg_type = struct.unpack("<i", hdr[12:16])[0]      # int32 (signed)
        
        # Read payload
        expected = response_len - 4  # subtract 4 bytes for msg_type
        payload_bytes = bytearray()
        while len(payload_bytes) < expected:
            chunk = self.client_socket.recv(expected - len(payload_bytes))
            if not chunk:
                raise RuntimeError("Connection closed while reading payload")
            payload_bytes.extend(chunk)
        
        return msg_type, bytes(payload_bytes)

    def parse_payload(self, payload: bytes) -> Dict[str, Any]:
        """
        Parse typed binary payload into structured data.
        
        Payload format (repeating):
        - Type tag (1 byte):
            1 = uint32 (4 bytes follow)
            2 = double/float64 (8 bytes follow)
            3 = text (null-terminated string follows)
            4 = int32 (4 bytes follow)
            6 = uint64 (8 bytes follow)
        
        Args:
            payload: Raw payload bytes (no VAST header).
        
        Returns:
            Dict with keys:
                "ints": List[int]       - int32 values
                "uints": List[int]      - uint32 values
                "doubles": List[float]  - float64 values
                "text": List[str]       - all text strings
                "last_text": str        - last text string (for single text results)
                "uint64s": List[int]    - uint64 values
        """
        ints: List[int] = []
        uints: List[int] = []
        doubles: List[float] = []
        texts: List[str] = []
        last_text: str = ""
        uint64s: List[int] = []
        
        p = 0
        n = len(payload)
        
        while p < n:
            if p + 1 > n:
                break
            
            type_tag = payload[p]
            
            if type_tag == 1:  # uint32
                if p + 5 > n:
                    break
                val = struct.unpack("<I", payload[p+1:p+5])[0]
                uints.append(val)
                p += 5
                
            elif type_tag == 2:  # double/float64
                if p + 9 > n:
                    break
                val = struct.unpack("<d", payload[p+1:p+9])[0]
                doubles.append(val)
                p += 9
                
            elif type_tag == 3:  # null-terminated text
                q = p + 1
                while q < n and payload[q] != 0:
                    q += 1
                if q >= n:
                    break  # no null terminator found
                text_bytes = payload[p+1:q]
                try:
                    text = text_bytes.decode("utf-8", errors="replace")
                except Exception:
                    text = "".join(chr(b) for b in text_bytes)
                last_text = text
                texts.append(text)
                p = q + 1  # skip null terminator
                
            elif type_tag == 4:  # int32
                if p + 5 > n:
                    break
                val = struct.unpack("<i", payload[p+1:p+5])[0]
                ints.append(val)
                p += 5
                
            elif type_tag == 6:  # uint64
                if p + 9 > n:
                    break
                val = struct.unpack("<Q", payload[p+1:p+9])[0]
                uint64s.append(val)
                p += 9
                
            else:
                # Unknown type tag, stop parsing
                break
        
        return {
            "ints": ints,
            "uints": uints,
            "doubles": doubles,
            "text": texts,
            "last_text": last_text,
            "uint64s": uint64s,
        }
    
    def bytes_from_int32(self, value: Union[int, List[int], np.ndarray]) -> bytes:
        """
        Convert int32 value(s) to typed bytes with type tag 4.
        Format: [type_tag=4, byte0, byte1, byte2, byte3] for each int32.
        """
        arr = np.atleast_1d(np.array(value, dtype='<i4'))  # explicit little-endian int32
        result = bytearray()
        for val in arr:
            result.append(4)  # type tag for int32
            result.extend(val.tobytes())
        return bytes(result)

    def bytes_from_uint32(self, value: Union[int, List[int], np.ndarray]) -> bytes:
        """
        Convert uint32 value(s) to typed bytes with type tag 1.
        Format: [type_tag=1, byte0, byte1, byte2, byte3] for each uint32.
        """
        arr = np.atleast_1d(np.array(value, dtype='<u4'))  # explicit little-endian uint32
        result = bytearray()
        for val in arr:
            result.append(1)  # type tag for uint32
            result.extend(val.tobytes())
        return bytes(result)

    def bytes_from_double(self, value: Union[float, List[float], np.ndarray]) -> bytes:
        """
        Convert float64 value(s) to typed bytes with type tag 2.
        Format: [type_tag=2, byte0, ..., byte7] for each double.
        """
        arr = np.atleast_1d(np.array(value, dtype='<f8'))  # explicit little-endian float64
        result = bytearray()
        for val in arr:
            result.append(2)  # type tag for double
            result.extend(val.tobytes())
        return bytes(result)

    def bytes_from_text(self, value: str) -> bytes:
        """
        Convert text to typed bytes with type tag 3.
        Format: [type_tag=3, utf8_bytes..., null_terminator=0]
        """
        text_bytes = value.encode('utf-8')
        return bytes([3]) + text_bytes + bytes([0])

    def bytes_from_data(self, value: Union[bytes, np.ndarray]) -> bytes:
        """
        Convert raw data to typed bytes with type tag 5.
        Format: [type_tag=5, len_byte0, len_byte1, len_byte2, len_byte3, data_bytes...]
        """
        if isinstance(value, np.ndarray):
            data_bytes = value.flatten().tobytes()
        else:
            data_bytes = bytes(value)
        
        length = np.uint32(len(data_bytes)).astype('<u4')  # explicit little-endian
        return bytes([5]) + length.tobytes() + data_bytes

    