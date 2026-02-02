"""
VASTControlClass - Python API for VAST Lite 1.5.0

This module provides a Python interface to the VAST (Volume Annotation and Segmentation Tool)
application via TCP socket communication.

Basic Usage:
    >>> from VCC import VASTControlClass
    >>> vcc = VASTControlClass()
    >>> if vcc.connect():
    >>>     info = vcc.get_info()
    >>>     print(f"Data size: {info['datasizex']} x {info['datasizey']} x {info['datasizez']}")
    >>>     vcc.disconnect()

Author: Daniel Berger (Harvard-Lichtman) - Python port from MATLAB
"""

import socket
import struct
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Default connection settings
HOST = "127.0.0.1"
PORT = 22081

# Error code descriptions
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
    """Python interface to VAST Lite 1.5.0 via TCP socket communication."""

    # ========================================================================
    # VAST API MESSAGE ID CONSTANTS
    # ========================================================================
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

    # ========================================================================
    # INITIALIZATION & CONNECTION
    # ========================================================================

    def __init__(self, host: str = HOST, port: int = PORT):
        """
        Initialize VAST Control Class.

        Args:
            host: VAST server hostname or IP address
            port: VAST server port
        """
        self.host = host
        self.port = port
        self.client: Optional[socket.socket] = None
        self.last_error = 0

    def connect(self, timeout: int = 100) -> bool:
        """
        Establish TCP connection to VAST server.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if successful, False if failed
        """
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(timeout)
            self.client.connect((self.host, self.port))
            self.last_error = 0
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None
            self.last_error = 1
            return False

    def disconnect(self) -> bool:
        """Close connection to VAST server."""
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
        return True

    def get_last_error(self) -> int:
        """Get the last error code."""
        return self.last_error

    def get_error_message(self, code: int = None) -> str:
        """Get error message for an error code."""
        if code is None:
            code = self.last_error
        return ERROR_CODES.get(code, f"Unknown error code: {code}")

    # ========================================================================
    # LOW-LEVEL COMMUNICATION
    # ========================================================================

    def _send_command(self, msg_id: int, payload: bytes = b"") -> Tuple[int, bytes]:
        """
        Send a command to VAST and receive the response.

        This is the core communication method that handles the full
        send/receive cycle efficiently in a single call.

        Args:
            msg_id: Message ID constant
            payload: Binary payload data

        Returns:
            Tuple of (msg_type, response_data)
            msg_type is 1 for success, other values indicate errors
        """
        if self.client is None:
            raise RuntimeError("Not connected to VAST API.")

        # Build and send message
        total_len = len(payload) + 4
        header = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", msg_id)
        self.client.sendall(header + payload)

        # Receive response header
        hdr = self.client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            raise RuntimeError(f"Invalid response header: {hdr!r}")

        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type = struct.unpack("<I", hdr[12:16])[0]

        # Receive payload
        expected = total_len - 4
        payload_bytes = bytearray()
        while len(payload_bytes) < expected:
            chunk = self.client.recv(expected - len(payload_bytes))
            if not chunk:
                raise RuntimeError("Incomplete payload from VAST.")
            payload_bytes.extend(chunk)

        return msg_type, bytes(payload_bytes)

    def _parse_payload(self, data: bytes) -> Dict[str, Any]:
        """
        Parse typed payload data from VAST response.

        Returns dict with keys: ints, uints, doubles, text, last_text, uint64s
        """
        ints, uints, doubles, texts, uint64s = [], [], [], [], []
        last_text = ""
        p, n = 0, len(data)

        while p < n:
            t = data[p]
            if t == 1:  # uint32
                if p + 5 > n: break
                uints.append(struct.unpack("<I", data[p+1:p+5])[0])
                p += 5
            elif t == 2:  # double
                if p + 9 > n: break
                doubles.append(struct.unpack("<d", data[p+1:p+9])[0])
                p += 9
            elif t == 3:  # null-terminated text
                q = p + 1
                while q < n and data[q] != 0:
                    q += 1
                if q >= n or data[q] != 0: break
                s = data[p+1:q].decode("utf-8", errors="replace")
                last_text = s
                texts.append(s)
                p = q + 1
            elif t == 4:  # int32
                if p + 5 > n: break
                ints.append(struct.unpack("<i", data[p+1:p+5])[0])
                p += 5
            elif t == 6:  # uint64
                if p + 9 > n: break
                uint64s.append(struct.unpack("<Q", data[p+1:p+9])[0])
                p += 9
            else:
                break

        return {"ints": ints, "uints": uints, "doubles": doubles,
                "text": texts, "last_text": last_text, "uint64s": uint64s}

    # ========================================================================
    # ENCODING HELPERS
    # ========================================================================

    @staticmethod
    def _encode_uint32(value: int) -> bytes:
        """Encode uint32 with type tag."""
        return b"\x01" + struct.pack("<I", value & 0xFFFFFFFF)

    @staticmethod
    def _encode_int32(value: int) -> bytes:
        """Encode int32 with type tag."""
        return b"\x04" + struct.pack("<i", value)

    @staticmethod
    def _encode_double(value: float) -> bytes:
        """Encode double with type tag."""
        return b"\x02" + struct.pack("<d", float(value))

    @staticmethod
    def _encode_text(text: str) -> bytes:
        """Encode null-terminated text with type tag."""
        return b"\x03" + text.encode("utf-8") + b"\x00"

    @staticmethod
    def _encode_data(data: np.ndarray) -> bytes:
        """Encode raw binary data with type tag and length."""
        raw = data.tobytes()
        return b"\x05" + struct.pack("<I", len(raw)) + raw

    # ========================================================================
    # STRUCT ENCODING/DECODING (for complex annotation data)
    # ========================================================================

    def _decode_to_struct(self, data: bytes) -> Tuple[Dict, int]:
        """Decode binary structured data to dictionary."""
        out, ptr = {}, 0
        n = len(data)

        try:
            while ptr < n:
                # Read variable name
                ptr2 = ptr
                while ptr2 < n and data[ptr2] != 0:
                    ptr2 += 1
                if ptr2 >= n: break
                name = data[ptr:ptr2].decode("utf-8", errors="replace")
                ptr = ptr2 + 1
                if ptr >= n: break

                dtype = data[ptr]
                ptr += 1

                if dtype == 0:  # uint32
                    if ptr + 4 > n: break
                    out[name] = struct.unpack("<I", data[ptr:ptr+4])[0]
                    ptr += 4
                elif dtype == 1:  # int32
                    if ptr + 4 > n: break
                    out[name] = struct.unpack("<i", data[ptr:ptr+4])[0]
                    ptr += 4
                elif dtype == 2:  # uint64
                    if ptr + 8 > n: break
                    out[name] = struct.unpack("<Q", data[ptr:ptr+8])[0]
                    ptr += 8
                elif dtype == 3:  # double
                    if ptr + 8 > n: break
                    out[name] = struct.unpack("<d", data[ptr:ptr+8])[0]
                    ptr += 8
                elif dtype == 4:  # string
                    ptr2 = ptr
                    while ptr2 < n and data[ptr2] != 0:
                        ptr2 += 1
                    if ptr2 >= n: break
                    out[name] = data[ptr:ptr2].decode("utf-8", errors="replace")
                    ptr = ptr2 + 1
                elif dtype in (5, 6, 7, 8):  # matrices
                    if ptr + 8 > n: break
                    xsize = struct.unpack("<I", data[ptr:ptr+4])[0]
                    ysize = struct.unpack("<I", data[ptr+4:ptr+8])[0]
                    ptr += 8
                    dtypes = {5: (np.uint32, 4), 6: (np.int32, 4),
                              7: (np.uint64, 8), 8: (np.float64, 8)}
                    dt, sz = dtypes[dtype]
                    nbytes = xsize * ysize * sz
                    if ptr + nbytes > n: break
                    mtx = np.frombuffer(data[ptr:ptr+nbytes], dtype=dt).reshape((ysize, xsize))
                    out[name] = mtx
                    ptr += nbytes
                elif dtype == 9:  # string array
                    if ptr + 8 > n: break
                    nstr = struct.unpack("<I", data[ptr:ptr+4])[0]
                    tlen = struct.unpack("<I", data[ptr+4:ptr+8])[0]
                    ptr += 8
                    if ptr + tlen > n: break
                    sdata = data[ptr:ptr+tlen]
                    ptr += tlen
                    strs, p = [], 0
                    for _ in range(nstr):
                        p2 = p
                        while p2 < len(sdata) and sdata[p2] != 0:
                            p2 += 1
                        if p2 < len(sdata):
                            strs.append(sdata[p:p2].decode("utf-8", errors="replace"))
                            p = p2 + 1
                    out[name] = strs
                else:
                    break
            return out, 1
        except Exception:
            return {}, 0

    def _encode_from_struct(self, data: Dict) -> Tuple[bytes, int]:
        """Encode dictionary to binary structured format."""
        try:
            out = bytearray()
            for key, value in data.items():
                kb = key.encode("utf-8")

                if isinstance(value, int) and not isinstance(value, bool):
                    if value >= 0 and value <= 0xFFFFFFFF:
                        out.extend(kb + b"\x00\x00" + struct.pack("<I", value))
                    elif value >= -2147483648 and value <= 2147483647:
                        out.extend(kb + b"\x00\x01" + struct.pack("<i", value))
                    else:
                        out.extend(kb + b"\x00\x02" + struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))
                elif isinstance(value, float):
                    out.extend(kb + b"\x00\x03" + struct.pack("<d", value))
                elif isinstance(value, str):
                    out.extend(kb + b"\x00\x04" + value.encode("utf-8") + b"\x00")
                elif isinstance(value, np.ndarray):
                    mtx = value.T if value.ndim >= 2 else np.atleast_2d(value).T
                    xs, ys = mtx.shape[0], mtx.shape[1] if mtx.ndim > 1 else 1
                    dtype_map = {np.uint32: 5, np.int32: 6, np.uint64: 7, np.float64: 8}
                    dt = dtype_map.get(value.dtype.type, 8)
                    out.extend(kb + bytes([0, dt]) + struct.pack("<II", xs, ys) + mtx.tobytes())
                elif isinstance(value, (list, tuple)):
                    cstr = b"".join(str(s).encode("utf-8") + b"\x00" for s in value)
                    out.extend(kb + b"\x00\x09" + struct.pack("<II", len(value), len(cstr)) + cstr)

            return bytes(out), 1
        except Exception:
            return b"", 0

    # ========================================================================
    # GENERAL INFO FUNCTIONS
    # ========================================================================

    def get_info(self) -> dict:
        """
        Get volume information from VAST.

        Returns dict with: datasizex/y/z, voxelsizex/y/z, cubesizex/y/z,
        currentviewx/y/z, nrofmiplevels. Empty dict on failure.
        """
        msg_type, data = self._send_command(self.GETINFO)

        if msg_type != 1:
            self.last_error = msg_type
            return {}

        p = self._parse_payload(data)
        if len(p["uints"]) == 7 and len(p["doubles"]) == 3 and len(p["ints"]) == 3:
            self.last_error = 0
            return {
                "datasizex": p["uints"][0], "datasizey": p["uints"][1], "datasizez": p["uints"][2],
                "voxelsizex": p["doubles"][0], "voxelsizey": p["doubles"][1], "voxelsizez": p["doubles"][2],
                "cubesizex": p["uints"][3], "cubesizey": p["uints"][4], "cubesizez": p["uints"][5],
                "currentviewx": p["ints"][0], "currentviewy": p["ints"][1], "currentviewz": p["ints"][2],
                "nrofmiplevels": p["uints"][6],
            }
        self.last_error = 2
        return {}

    def get_api_version(self) -> int:
        """Get VAST API version number."""
        msg_type, data = self._send_command(self.GETAPIVERSION)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        try:
            return struct.unpack("<I", data[1:5])[0]
        except:
            self.last_error = 2
            return 0

    def get_hardware_info(self) -> dict:
        """Get hardware information from VAST."""
        msg_type, data = self._send_command(self.GETHARDWAREINFO)
        if msg_type != 1:
            self.last_error = msg_type
            return {}

        p = self._parse_payload(data)
        if len(p["uints"]) == 1 and len(p["doubles"]) == 7 and len(p["text"]) == 5:
            self.last_error = 0
            return {
                "computername": p["text"][0],
                "processorname": p["text"][1],
                "processorspeed_ghz": p["doubles"][0],
                "nrofprocessorcores": p["uints"][0],
                "tickspeedmhz": p["doubles"][1],
                "mmxssecapabilities": p["text"][2],
                "totalmemorygb": p["doubles"][2],
                "freememorygb": p["doubles"][3],
                "graphicscardname": p["text"][3],
                "graphicsdedicatedvideomemgb": p["doubles"][4],
                "graphicsdedicatedsysmemgb": p["doubles"][5],
                "graphicssharedsysmemgb": p["doubles"][6],
                "graphicsrasterizerused": p["text"][4],
            }
        self.last_error = 2
        return {}

    # ========================================================================
    # LAYER FUNCTIONS
    # ========================================================================

    def get_number_of_layers(self) -> int:
        """Get number of layers."""
        msg_type, data = self._send_command(self.GETNROFLAYERS)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def get_layer_info(self, layer_nr: int) -> dict:
        """Get information about a specific layer."""
        msg_type, data = self._send_command(self.GETLAYERINFO, self._encode_uint32(layer_nr))
        if msg_type != 1:
            self.last_error = msg_type
            return {}

        p = self._parse_payload(data)
        if len(p["ints"]) == 8 and len(p["uints"]) == 7 and len(p["doubles"]) == 3:
            type_map = {0: "Image", 1: "Segmentation", 2: "Annotation",
                        3: "VSVR image", 6: "VSVI image", 7: "Tool"}
            self.last_error = 0
            return {
                "type": type_map.get(p["ints"][0], "Other"),
                "type_id": p["ints"][0],
                "editable": p["ints"][1],
                "visible": p["ints"][2],
                "brightness": p["ints"][3],
                "contrast": p["ints"][4],
                "opacitylevel": p["doubles"][0],
                "brightnesslevel": p["doubles"][1],
                "contrastlevel": p["doubles"][2],
                "blendmode": p["ints"][5],
                "blendoradd": p["ints"][6],
                "tintcolor": p["uints"][0],
                "name": p["last_text"],
                "redtargetcolor": p["uints"][1],
                "greentargetcolor": p["uints"][2],
                "bluetargetcolor": p["uints"][3],
                "bytesperpixel": p["uints"][4],
                "ischanged": p["ints"][7],
                "inverted": p["uints"][5],
                "solomode": p["uints"][6],
            }
        self.last_error = 2
        return {}

    def set_layer_info(self, layer_nr: int, info: dict) -> bool:
        """Set layer information. Only provided fields are updated."""
        if not info:
            return True

        xflags = 0
        msg = self._encode_uint32(layer_nr)

        fields = [
            ("editable", 1, self._encode_int32),
            ("visible", 2, self._encode_int32),
            ("brightness", 4, self._encode_int32),
            ("contrast", 8, self._encode_int32),
            ("opacitylevel", 16, self._encode_double),
            ("brightnesslevel", 32, self._encode_double),
            ("contrastlevel", 64, self._encode_double),
            ("blendmode", 128, self._encode_int32),
            ("blendoradd", 256, self._encode_int32),
            ("tintcolor", 512, self._encode_uint32),
            ("redtargetcolor", 1024, self._encode_uint32),
            ("greentargetcolor", 2048, self._encode_uint32),
            ("bluetargetcolor", 4096, self._encode_uint32),
            ("inverted", 8192, self._encode_uint32),
            ("solomode", 16384, self._encode_uint32),
        ]

        for name, flag, encoder in fields:
            if name in info and info[name] is not None:
                msg += encoder(info[name])
                xflags |= flag

        if xflags == 0:
            return True

        payload = self._encode_uint32(xflags) + msg
        msg_type, _ = self._send_command(self.SETLAYERINFO, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_layer_nr(self) -> dict:
        """Get selected layer numbers."""
        msg_type, data = self._send_command(self.GETSELECTEDLAYERNR)
        if msg_type != 1:
            self.last_error = msg_type
            return {}

        p = self._parse_payload(data)
        ints = p["ints"]
        if len(ints) == 3:
            self.last_error = 0
            return {"selected_layer": ints[0], "selected_em_layer": ints[1],
                    "selected_segment_layer": ints[2]}
        elif len(ints) == 5:
            self.last_error = 0
            return {"selected_layer": ints[0], "selected_em_layer": ints[1],
                    "selected_anno_layer": ints[2], "selected_segment_layer": ints[3],
                    "selected_tool_layer": ints[4]}
        self.last_error = 2
        return {}

    def set_selected_layer_nr(self, layer_nr: int) -> bool:
        """Set selected layer."""
        msg_type, _ = self._send_command(self.SETSELECTEDLAYERNR, self._encode_uint32(layer_nr))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def add_new_layer(self, layer_type: int, name: str, ref_id: int = -1) -> int:
        """Add a new layer. Returns new layer number or 0 on failure."""
        payload = self._encode_int32(layer_type) + self._encode_int32(ref_id) + self._encode_text(name)
        msg_type, data = self._send_command(self.ADDNEWLAYER, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def load_layer(self, filename: str, ref_id: int = -1) -> int:
        """Load a layer from file. Returns layer number or 0 on failure."""
        payload = self._encode_int32(ref_id) + self._encode_text(filename)
        msg_type, data = self._send_command(self.LOADLAYER, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def save_layer(self, layer_nr: int, filename: str, force: bool = False, subformat: int = 0) -> bool:
        """Save a layer to file."""
        payload = (self._encode_uint32(layer_nr) + self._encode_uint32(int(force)) +
                   self._encode_uint32(subformat) + self._encode_text(filename))
        msg_type, _ = self._send_command(self.SAVELAYER, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def remove_layer(self, layer_nr: int, force: bool = False) -> bool:
        """Remove a layer."""
        payload = self._encode_uint32(layer_nr) + self._encode_uint32(int(force))
        msg_type, _ = self._send_command(self.REMOVELAYER, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def move_layer(self, moved_layer_nr: int, after_layer_nr: int) -> int:
        """Move layer position. Returns new layer number or 0 on failure."""
        payload = self._encode_uint32(moved_layer_nr) + self._encode_uint32(after_layer_nr)
        msg_type, data = self._send_command(self.MOVELAYER, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def get_mipmap_scale_factors(self, layer_nr: int) -> List[List[int]]:
        """Get mipmap scale factors for a layer."""
        msg_type, data = self._send_command(self.GETMIPMAPSCALEFACTORS, self._encode_uint32(layer_nr))
        if msg_type != 1 or len(data) < 4 or len(data) % 4 != 0:
            self.last_error = 2 if msg_type == 1 else msg_type
            return []

        uid = struct.unpack(f"<{len(data)//4}I", data)
        num_levels = uid[0]
        if 1 + 3 * num_levels > len(uid):
            self.last_error = 2
            return []

        matrix = []
        idx = 1
        for _ in range(num_levels):
            matrix.append([uid[idx], uid[idx + 1], uid[idx + 2]])
            idx += 3

        # Skip first row (mip level 0 is always [1,1,1])
        self.last_error = 0
        return matrix[1:] if matrix else []

    def get_data_size_at_mip(self, layer_nr: int, mip: int) -> Optional[Tuple[int, int, int]]:
        """Get data size at a specific mip level."""
        info = self.get_info()
        if not info:
            return None

        if mip == 0:
            return (info["datasizex"], info["datasizey"], info["datasizez"])

        msf = self.get_mipmap_scale_factors(layer_nr)
        if not msf or mip > len(msf):
            return None

        return (int(info["datasizex"] / msf[mip - 1][0]),
                int(info["datasizey"] / msf[mip - 1][1]),
                int(info["datasizez"] / msf[mip - 1][2]))

    # ========================================================================
    # API LAYER CONTROL
    # ========================================================================

    def get_api_layers_enabled(self) -> int:
        """Check if separate API layer selection is enabled. Returns 1/0 or -1 on error."""
        msg_type, data = self._send_command(self.GETAPILAYERSENABLED)
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return -1

    def set_api_layers_enabled(self, enabled: bool) -> bool:
        """Enable or disable separate API layer selection."""
        msg_type, _ = self._send_command(self.SETAPILAYERSENABLED, self._encode_uint32(int(enabled)))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_api_layer_nr(self) -> dict:
        """Get selected API layer numbers."""
        msg_type, data = self._send_command(self.GETSELECTEDAPILAYERNR)
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        p = self._parse_payload(data)
        if len(p["ints"]) == 5:
            self.last_error = 0
            return {
                "selected_layer": p["ints"][0],
                "selected_em_layer": p["ints"][1],
                "selected_anno_layer": p["ints"][2],
                "selected_segment_layer": p["ints"][3],
                "selected_tool_layer": p["ints"][4],
            }
        self.last_error = 2
        return {}

    def set_selected_api_layer_nr(self, layer_nr: int) -> bool:
        """Select a layer for API access."""
        msg_type, _ = self._send_command(self.SETSELECTEDAPILAYERNR, self._encode_int32(layer_nr))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    # ========================================================================
    # SEGMENT FUNCTIONS
    # ========================================================================

    def get_number_of_segments(self) -> int:
        """Get number of segments."""
        msg_type, data = self._send_command(self.GETNUMBEROFSEGMENTS)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def get_segment_data(self, segment_id: int) -> dict:
        """Get metadata for a specific segment."""
        msg_type, data = self._send_command(self.GETSEGMENTDATA, self._encode_uint32(segment_id))
        if msg_type != 1:
            self.last_error = msg_type
            return {}

        p = self._parse_payload(data)
        if len(p["ints"]) == 9 and len(p["uints"]) == 9:
            self.last_error = 0
            return {
                "id": p["uints"][0], "flags": p["uints"][1],
                "col1": p["uints"][2], "col2": p["uints"][3],
                "anchorpoint": p["ints"][0:3],
                "hierarchy": list(p["uints"][4:8]),
                "collapsednr": p["uints"][8],
                "boundingbox": p["ints"][3:9],
            }
        self.last_error = 2
        return {}

    def get_segment_name(self, segment_id: int) -> str:
        """Get segment name."""
        msg_type, data = self._send_command(self.GETSEGMENTNAME, self._encode_uint32(segment_id))
        if msg_type != 1:
            self.last_error = msg_type
            return ""
        p = self._parse_payload(data)
        self.last_error = 0
        return p["last_text"]

    def set_segment_name(self, segment_id: int, name: str) -> bool:
        """Set segment name."""
        payload = self._encode_uint32(segment_id) + self._encode_text(name)
        msg_type, _ = self._send_command(self.SETSEGMENTNAME, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_anchor_point(self, segment_id: int, x: int, y: int, z: int) -> bool:
        """Set segment anchor point."""
        payload = (self._encode_uint32(segment_id) + self._encode_uint32(x) +
                   self._encode_uint32(y) + self._encode_uint32(z))
        msg_type, _ = self._send_command(self.SETANCHORPOINT, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_segment_color_8(self, segment_id: int, r1: int, g1: int, b1: int, p1: int,
                            r2: int, g2: int, b2: int, p2: int) -> bool:
        """Set segment colors using 8-bit RGB values and patterns."""
        v1 = (p1 & 0xFF) | ((b1 & 0xFF) << 8) | ((g1 & 0xFF) << 16) | ((r1 & 0xFF) << 24)
        v2 = (p2 & 0xFF) | ((b2 & 0xFF) << 8) | ((g2 & 0xFF) << 16) | ((r2 & 0xFF) << 24)
        payload = self._encode_uint32(segment_id) + self._encode_uint32(v1) + self._encode_uint32(v2)
        msg_type, _ = self._send_command(self.SETSEGMENTCOLOR, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_segment_color_32(self, segment_id: int, col1: int, col2: int) -> bool:
        """Set segment colors using 32-bit values."""
        payload = self._encode_uint32(segment_id) + self._encode_uint32(col1) + self._encode_uint32(col2)
        msg_type, _ = self._send_command(self.SETSEGMENTCOLOR, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_segment_bbox(self, segment_id: int, minx: int, maxx: int,
                         miny: int, maxy: int, minz: int, maxz: int) -> bool:
        """Set segment bounding box."""
        payload = (self._encode_uint32(segment_id) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz))
        msg_type, _ = self._send_command(self.SETSEGMENTBBOX, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_all_segment_data(self) -> List[dict]:
        """Get metadata for all segments."""
        msg_type, data = self._send_command(self.GETALLSEGMENTDATA)
        if msg_type != 1 or len(data) < 4:
            self.last_error = msg_type if msg_type != 1 else 2
            return []

        arr = np.frombuffer(data, dtype=np.uint32)
        iarr = np.frombuffer(data, dtype=np.int32)
        num_segments = arr[0]
        segments = []
        sp = 1

        for i in range(num_segments):
            segments.append({
                "id": i, "flags": int(arr[sp]), "col1": int(arr[sp + 1]), "col2": int(arr[sp + 2]),
                "anchorpoint": [int(iarr[sp + 3]), int(iarr[sp + 4]), int(iarr[sp + 5])],
                "hierarchy": [int(arr[sp + 6]), int(arr[sp + 7]), int(arr[sp + 8]), int(arr[sp + 9])],
                "collapsednr": int(arr[sp + 10]),
                "boundingbox": [int(iarr[sp + 11]), int(iarr[sp + 12]), int(iarr[sp + 13]),
                                int(iarr[sp + 14]), int(iarr[sp + 15]), int(iarr[sp + 16])],
            })
            sp += 17

        self.last_error = 0
        return segments

    def get_all_segment_data_matrix(self) -> Tuple[np.ndarray, int]:
        """Get all segment data as a matrix. Returns (matrix, success)."""
        msg_type, data = self._send_command(self.GETALLSEGMENTDATA)
        if msg_type != 1 or len(data) < 4:
            self.last_error = msg_type if msg_type != 1 else 2
            return np.array([]), 0

        uid = np.frombuffer(data, dtype=np.uint32)
        sid = np.frombuffer(data, dtype=np.int32)
        num_segments = uid[0]

        if num_segments == 0:
            self.last_error = 0
            return np.zeros((0, 24), dtype=np.float64), 1

        matrix = np.zeros((num_segments, 24), dtype=np.float64)
        sp = 1

        for i in range(num_segments):
            matrix[i, 0] = i
            matrix[i, 1] = uid[sp]
            matrix[i, 2] = (uid[sp + 1] >> 24) & 0xFF  # R
            matrix[i, 3] = (uid[sp + 1] >> 16) & 0xFF  # G
            matrix[i, 4] = (uid[sp + 1] >> 8) & 0xFF   # B
            matrix[i, 5] = uid[sp + 1] & 0xFF          # P
            matrix[i, 6] = (uid[sp + 2] >> 24) & 0xFF
            matrix[i, 7] = (uid[sp + 2] >> 16) & 0xFF
            matrix[i, 8] = (uid[sp + 2] >> 8) & 0xFF
            matrix[i, 9] = uid[sp + 2] & 0xFF
            matrix[i, 10:13] = sid[sp + 3:sp + 6]
            matrix[i, 13:17] = uid[sp + 6:sp + 10]
            matrix[i, 17] = uid[sp + 10]
            matrix[i, 18:24] = sid[sp + 11:sp + 17]
            sp += 17

        self.last_error = 0
        return matrix[1:, :] if matrix.shape[0] > 0 else matrix, 1

    def get_all_segment_names(self) -> List[str]:
        """Get names of all segments."""
        msg_type, data = self._send_command(self.GETALLSEGMENTNAMES)
        if msg_type != 1 or len(data) < 4:
            self.last_error = msg_type if msg_type != 1 else 2
            return []

        num_names = struct.unpack("<I", data[0:4])[0]
        names, pos = [], 4

        for _ in range(num_names):
            end = pos
            while end < len(data) and data[end] != 0:
                end += 1
            if end >= len(data):
                self.last_error = 2
                return []
            names.append(data[pos:end].decode("utf-8", errors="replace"))
            pos = end + 1

        self.last_error = 0
        return names

    def get_selected_segment_nr(self) -> int:
        """Get currently selected segment. Returns -1 on failure."""
        msg_type, data = self._send_command(self.GETSELECTEDSEGMENTNR)
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return -1

    def set_selected_segment_nr(self, segment_id: int) -> bool:
        """Select a segment."""
        msg_type, _ = self._send_command(self.SETSELECTEDSEGMENTNR, self._encode_int32(segment_id))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_first_segment_nr(self) -> int:
        """Get first segment number. Returns -1 on failure."""
        msg_type, data = self._send_command(self.GETFIRSTSEGMENTNR)
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return -1

    def add_segment(self, ref_id: int, next_or_child: int, name: str) -> int:
        """Add a new segment. Returns new ID or 0 on failure."""
        payload = self._encode_uint32(ref_id) + self._encode_uint32(next_or_child) + self._encode_text(name)
        msg_type, data = self._send_command(self.ADDSEGMENT, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def move_segment(self, segment_id: int, ref_id: int, next_or_child: int) -> bool:
        """Move a segment in hierarchy."""
        payload = (self._encode_uint32(segment_id) + self._encode_uint32(ref_id) +
                   self._encode_uint32(next_or_child))
        msg_type, _ = self._send_command(self.MOVESEGMENT, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    # ========================================================================
    # VIEW FUNCTIONS
    # ========================================================================

    def get_view_coordinates(self) -> Tuple[int, int, int]:
        """Get current view coordinates. Returns (0,0,0) on failure."""
        msg_type, data = self._send_command(self.GETVIEWCOORDINATES)
        if msg_type != 1:
            self.last_error = msg_type
            return 0, 0, 0
        p = self._parse_payload(data)
        if len(p["ints"]) == 3:
            self.last_error = 0
            return p["ints"][0], p["ints"][1], p["ints"][2]
        self.last_error = 2
        return 0, 0, 0

    def set_view_coordinates(self, x: int, y: int, z: int) -> bool:
        """Set view coordinates."""
        payload = self._encode_uint32(x) + self._encode_uint32(y) + self._encode_uint32(z)
        msg_type, _ = self._send_command(self.SETVIEWCOORDINATES, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_view_zoom(self) -> int:
        """Get current zoom level."""
        msg_type, data = self._send_command(self.GETVIEWZOOM)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["ints"]) == 1:
            self.last_error = 0
            return p["ints"][0]
        self.last_error = 2
        return 0

    def set_view_zoom(self, zoom: int) -> bool:
        """Set zoom level."""
        msg_type, _ = self._send_command(self.SETVIEWZOOM, self._encode_int32(zoom))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def refresh_layer_region(self, layer_nr: int, minx: int, maxx: int,
                             miny: int, maxy: int, minz: int, maxz: int) -> bool:
        """Refresh a region of a layer in the display."""
        payload = (self._encode_uint32(layer_nr) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz))
        msg_type, _ = self._send_command(self.REFRESHLAYERREGION, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    # ========================================================================
    # SEGMENTATION IMAGE FUNCTIONS
    # ========================================================================

    def get_seg_image_raw(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, immediate: bool = False,
                          request_load: bool = False) -> Optional[bytes]:
        """Get raw segmentation image data as bytes (uint16 little-endian)."""
        cmd = self.GETSEGIMAGERAWIMMEDIATE if immediate else self.GETSEGIMAGERAW
        payload = (self._encode_uint32(miplevel) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz))
        if immediate:
            payload += self._encode_uint32(int(request_load))

        msg_type, data = self._send_command(cmd, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return None
        self.last_error = 0
        return data

    def get_seg_image_rle(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, surf_only: bool = False,
                          immediate: bool = False, request_load: bool = False) -> Optional[bytes]:
        """Get RLE-encoded segmentation image data."""
        if immediate:
            cmd = self.GETSEGIMAGERLEIMMEDIATE
        else:
            cmd = self.GETSEGIMAGESURFRLE if surf_only else self.GETSEGIMAGERLE

        payload = (self._encode_uint32(miplevel) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz))
        if immediate:
            payload += self._encode_uint32(int(request_load))

        msg_type, data = self._send_command(cmd, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return None
        self.last_error = 0
        return data

    def get_seg_image_rle_decoded(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                   minz: int, maxz: int, surf_only: bool = False,
                                   immediate: bool = False, request_load: bool = False) -> Optional[np.ndarray]:
        """Get decoded segmentation image from RLE. Returns array of shape (Y, X, Z)."""
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                          surf_only, immediate, request_load)
        if rle_data is None:
            return None

        rle = np.frombuffer(rle_data, dtype=np.uint16)
        size_x, size_y, size_z = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1

        result = np.zeros(size_x * size_y * size_z, dtype=np.uint16)
        dp = 0
        for sp in range(0, len(rle), 2):
            val, count = rle[sp], rle[sp + 1]
            result[dp:dp + count] = val
            dp += count

        return np.transpose(result.reshape(size_x, size_y, size_z), (1, 0, 2))

    def get_rle_count_unique(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                             minz: int, maxz: int, surf_only: bool = False,
                             immediate: bool = False, request_load: bool = False) -> Tuple[List[int], List[int]]:
        """Get unique segment IDs and counts from RLE without full decode."""
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                          surf_only, immediate, request_load)
        if rle_data is None:
            return [], []

        rle = np.frombuffer(rle_data, dtype=np.uint16)
        if len(rle) == 0:
            return [], []

        max_val = int(np.max(rle[0::2]))
        counts = np.zeros(max_val + 1, dtype=np.uint64)
        for i in range(0, len(rle), 2):
            counts[rle[i]] += rle[i + 1]

        nonzero = np.where(counts > 0)[0]
        return nonzero.tolist(), counts[nonzero].tolist()

    def get_seg_image_rle_decoded_count_unique(self, miplevel: int, minx: int, maxx: int,
                                                miny: int, maxy: int, minz: int, maxz: int,
                                                surf_only: bool = False, immediate: bool = False,
                                                request_load: bool = False) -> Tuple[Optional[np.ndarray], List[int], List[int]]:
        """Get decoded image and unique counts together."""
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                          surf_only, immediate, request_load)
        if rle_data is None:
            return None, [], []

        rle = np.frombuffer(rle_data, dtype=np.uint16)
        size_x, size_y, size_z = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1

        max_val = int(np.max(rle[0::2]))
        counts = np.zeros(max_val + 1, dtype=np.uint64)
        result = np.zeros(size_x * size_y * size_z, dtype=np.uint16)
        dp = 0

        for sp in range(0, len(rle), 2):
            val, count = rle[sp], rle[sp + 1]
            result[dp:dp + count] = val
            counts[val] += count
            dp += count

        img = np.transpose(result.reshape(size_x, size_y, size_z), (1, 0, 2))
        nonzero = np.where(counts > 0)[0]
        return img, nonzero.tolist(), counts[nonzero].tolist()

    def get_seg_image_rle_decoded_bboxes(self, miplevel: int, minx: int, maxx: int,
                                          miny: int, maxy: int, minz: int, maxz: int,
                                          surf_only: bool = False, immediate: bool = False,
                                          request_load: bool = False) -> Tuple[Optional[np.ndarray], List[int], List[int], List[List[int]]]:
        """Get decoded image, unique counts, and bounding boxes."""
        rle_data = self.get_seg_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz,
                                          surf_only, immediate, request_load)
        if rle_data is None:
            return None, [], [], []

        rle = np.frombuffer(rle_data, dtype=np.uint16)
        sx, sy, sz = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1

        max_val = int(np.max(rle[0::2]))
        counts = np.zeros(max_val + 1, dtype=np.uint64)
        bboxes = np.full((max_val + 1, 6), -1, dtype=np.int32)
        result = np.zeros(sx * sy * sz, dtype=np.uint16)
        dp = 0

        for sp in range(0, len(rle), 2):
            val, count = int(rle[sp]), int(rle[sp + 1])
            result[dp:dp + count] = val
            counts[val] += count

            # Compute bbox for this run
            z1, r = divmod(dp, sx * sy)
            y1, x1 = divmod(r, sx)
            z2, r = divmod(dp + count - 1, sx * sy)
            y2, x2 = divmod(r, sx)

            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            zmin, zmax = z1, z2
            if zmax > zmin:
                xmin, xmax, ymin, ymax = 0, sx - 1, 0, sy - 1
            elif ymax > ymin:
                xmin, xmax = 0, sx - 1

            if bboxes[val, 0] == -1:
                bboxes[val] = [xmin, ymin, zmin, xmax, ymax, zmax]
            else:
                bboxes[val, 0] = min(bboxes[val, 0], xmin)
                bboxes[val, 1] = min(bboxes[val, 1], ymin)
                bboxes[val, 2] = min(bboxes[val, 2], zmin)
                bboxes[val, 3] = max(bboxes[val, 3], xmax)
                bboxes[val, 4] = max(bboxes[val, 4], ymax)
                bboxes[val, 5] = max(bboxes[val, 5], zmax)
            dp += count

        img = result.reshape(sx, sy, sz)
        nonzero = np.where(counts > 0)[0]
        return img, nonzero.tolist(), counts[nonzero].tolist(), bboxes[nonzero].tolist()

    def set_seg_translation(self, source: List[int], target: List[int]) -> bool:
        """Set segment ID translation table for image retrieval."""
        if len(source) != len(target):
            self.last_error = 50
            return False

        payload = b""
        for s, t in zip(source, target):
            payload += self._encode_uint32(s) + self._encode_uint32(t)

        msg_type, _ = self._send_command(self.SETSEGTRANSLATION, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_seg_image_raw(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, seg_image: np.ndarray) -> bool:
        """Write raw segmentation image data. Image shape should be (Y, X, Z)."""
        expected = (maxy - miny + 1, maxx - minx + 1, maxz - minz + 1)
        if seg_image.shape != expected:
            self.last_error = 13
            return False

        seg_transposed = np.transpose(seg_image, (1, 0, 2)).astype(np.uint16)
        seg_bytes = seg_transposed.flatten().tobytes()

        params = (self._encode_uint32(miplevel) +
                  self._encode_uint32(minx) + self._encode_uint32(maxx) +
                  self._encode_uint32(miny) + self._encode_uint32(maxy) +
                  self._encode_uint32(minz) + self._encode_uint32(maxz))
        payload = params + b"\x05" + struct.pack("<I", len(seg_bytes)) + seg_bytes

        msg_type, _ = self._send_command(self.SETSEGIMAGERAW, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_seg_image_rle(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                          minz: int, maxz: int, seg_image: np.ndarray) -> bool:
        """Write RLE-encoded segmentation image. Falls back to raw if RLE is larger."""
        expected = (maxy - miny + 1, maxx - minx + 1, maxz - minz + 1)
        if seg_image.shape != expected:
            self.last_error = 13
            return False

        seg_flat = np.transpose(seg_image, (1, 0, 2)).astype(np.uint16).flatten()

        # RLE encode
        rle = []
        if len(seg_flat) > 0:
            val, count = seg_flat[0], 1
            for i in range(1, len(seg_flat)):
                if seg_flat[i] == val and count < 65535:
                    count += 1
                else:
                    rle.extend([val, count])
                    val, count = seg_flat[i], 1
            rle.extend([val, count])

        rle_arr = np.array(rle, dtype=np.uint16)
        if len(rle_arr) >= len(seg_flat):
            return self.set_seg_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz, seg_image)

        params = (self._encode_uint32(miplevel) +
                  self._encode_uint32(minx) + self._encode_uint32(maxx) +
                  self._encode_uint32(miny) + self._encode_uint32(maxy) +
                  self._encode_uint32(minz) + self._encode_uint32(maxz))
        rle_bytes = rle_arr.tobytes()
        payload = params + b"\x05" + struct.pack("<I", len(rle_bytes)) + rle_bytes

        msg_type, _ = self._send_command(self.SETSEGIMAGERLE, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    # ========================================================================
    # EM IMAGE FUNCTIONS
    # ========================================================================

    def get_em_image_raw(self, layer_nr: int, miplevel: int, minx: int, maxx: int,
                         miny: int, maxy: int, minz: int, maxz: int,
                         immediate: bool = False, request_load: bool = False) -> Optional[bytes]:
        """Get raw EM/image layer data."""
        cmd = self.GETEMIMAGERAWIMMEDIATE if immediate else self.GETEMIMAGERAW
        payload = (self._encode_uint32(layer_nr) + self._encode_uint32(miplevel) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz))
        if immediate:
            payload += self._encode_uint32(int(request_load))

        msg_type, data = self._send_command(cmd, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return None
        self.last_error = 0
        return data

    def get_em_image(self, layer_nr: int, miplevel: int, minx: int, maxx: int,
                     miny: int, maxy: int, minz: int, maxz: int,
                     immediate: bool = False, request_load: bool = False) -> Optional[np.ndarray]:
        """Get EM image as numpy array with proper reshaping."""
        raw = self.get_em_image_raw(layer_nr, miplevel, minx, maxx, miny, maxy, minz, maxz,
                                    immediate, request_load)
        if raw is None:
            return None

        sx, sy, sz = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        bpp = len(raw) // (sx * sy * sz)

        if bpp == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(sx, sy, sz)
            return np.transpose(arr, (1, 0, 2)) if sz > 1 else arr.reshape(sx, sy).T
        elif bpp == 3:
            arr = np.frombuffer(raw, dtype=np.uint8)
            if sz == 1:
                return np.flip(np.transpose(arr.reshape(3, sx, sy), (2, 1, 0)), axis=2)
            return np.flip(np.transpose(arr.reshape(3, sx, sy, sz), (2, 1, 3, 0)), axis=3)
        elif bpp == 4:
            arr = np.frombuffer(raw, dtype=np.uint32).reshape(sx, sy, sz)
            return np.transpose(arr, (1, 0, 2)) if sz > 1 else arr.reshape(sx, sy).T
        elif bpp == 8:
            arr = np.frombuffer(raw, dtype=np.uint64).reshape(sx, sy, sz)
            return np.transpose(arr, (1, 0, 2)) if sz > 1 else arr.reshape(sx, sy).T

        self.last_error = 2
        return None

    def get_pixel_value(self, layer_nr: int, miplevel: int, x: int, y: int, z: int) -> int:
        """Get pixel value at coordinates."""
        payload = (self._encode_uint32(layer_nr) + self._encode_uint32(miplevel) +
                   self._encode_uint32(x) + self._encode_uint32(y) + self._encode_uint32(z))
        msg_type, data = self._send_command(self.GETPIXELVALUE, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uint64s"]) == 1:
            self.last_error = 0
            return p["uint64s"][0]
        self.last_error = 2
        return 0

    # ========================================================================
    # SCREENSHOT FUNCTIONS
    # ========================================================================

    def get_screenshot_image_raw(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                  minz: int, maxz: int, collapse_seg: bool = False) -> Optional[bytes]:
        """Get raw screenshot image."""
        payload = (self._encode_uint32(miplevel) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz) +
                   self._encode_uint32(int(collapse_seg)))
        msg_type, data = self._send_command(self.GETSCREENSHOTIMAGERAW, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return None
        self.last_error = 0
        return data

    def get_screenshot_image(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                              minz: int, maxz: int, collapse_seg: bool = False,
                              use_rle: bool = False) -> Optional[np.ndarray]:
        """Get screenshot as numpy array."""
        if use_rle:
            raw = self.get_screenshot_image_rle(miplevel, minx, maxx, miny, maxy, minz, maxz, collapse_seg)
        else:
            raw = self.get_screenshot_image_raw(miplevel, minx, maxx, miny, maxy, minz, maxz, collapse_seg)

        if raw is None:
            return None

        sx, sy, sz = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        arr = np.frombuffer(raw, dtype=np.uint8)

        if sz == 1:
            return np.flip(np.transpose(arr.reshape(3, sx, sy), (2, 1, 0)), axis=2)
        return np.flip(np.transpose(arr.reshape(3, sx, sy, sz), (2, 1, 3, 0)), axis=3)

    def get_screenshot_image_rle(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                  minz: int, maxz: int, collapse_seg: bool = False) -> Optional[bytes]:
        """Get RLE-encoded screenshot."""
        payload = (self._encode_uint32(miplevel) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz) +
                   self._encode_uint32(int(collapse_seg)))
        msg_type, data = self._send_command(self.GETSCREENSHOTIMAGERLE, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return None

        expected = (maxx - minx + 1) * (maxy - miny + 1) * (maxz - minz + 1) * 3
        if len(data) == expected:
            self.last_error = 0
            return data

        # Decode RLE
        rle = np.frombuffer(data, dtype=np.uint8)
        decoded = np.zeros(expected, dtype=np.uint8)
        dp = 0
        for sp in range(0, len(rle) - 3, 4):
            r, g, b, count = rle[sp], rle[sp+1], rle[sp+2], rle[sp+3]
            for _ in range(count):
                if dp + 2 < len(decoded):
                    decoded[dp], decoded[dp+1], decoded[dp+2] = r, g, b
                    dp += 3
        self.last_error = 0
        return decoded.tobytes()

    def order_screenshot_image(self, miplevel: int, minx: int, maxx: int, miny: int, maxy: int,
                                minz: int, maxz: int, collapse_seg: bool = False) -> bool:
        """Order screenshot (async). Call pickup_screenshot_image() to receive."""
        if self.client is None:
            return False

        payload = (self._encode_uint32(miplevel) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz) +
                   self._encode_uint32(int(collapse_seg)))

        total_len = len(payload) + 4
        header = b"VAST" + struct.pack("<Q", total_len) + struct.pack("<I", self.GETSCREENSHOTIMAGERAW)
        self.client.sendall(header + payload)
        return True

    def pickup_screenshot_image(self, minx: int, maxx: int, miny: int, maxy: int,
                                 minz: int, maxz: int) -> Optional[np.ndarray]:
        """Receive screenshot after order_screenshot_image()."""
        if self.client is None:
            return None

        hdr = self.client.recv(16)
        if len(hdr) < 16 or not hdr.startswith(b"VAST"):
            self.last_error = 2
            return None

        total_len = struct.unpack("<Q", hdr[4:12])[0]
        msg_type = struct.unpack("<I", hdr[12:16])[0]
        if msg_type != 1:
            self.last_error = msg_type
            return None

        expected = total_len - 4
        payload = bytearray()
        while len(payload) < expected:
            chunk = self.client.recv(expected - len(payload))
            if not chunk:
                self.last_error = 2
                return None
            payload.extend(chunk)

        sx, sy, sz = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        arr = np.frombuffer(bytes(payload), dtype=np.uint8)

        if sz == 1:
            result = np.flip(np.transpose(arr.reshape(3, sx, sy), (2, 1, 0)), axis=2)
        else:
            result = np.flip(np.transpose(arr.reshape(3, sx, sy, sz), (2, 1, 3, 0)), axis=3)

        self.last_error = 0
        return result

    # ========================================================================
    # ANNOTATION OBJECT FUNCTIONS
    # ========================================================================

    def get_anno_layer_nr_of_objects(self) -> Tuple[int, int]:
        """Get number of annotation objects and first object number."""
        msg_type, data = self._send_command(self.GETANNOLAYERNROFOBJECTS)
        if msg_type != 1:
            self.last_error = msg_type
            return 0, -1
        p = self._parse_payload(data)
        if len(p["uints"]) == 2:
            self.last_error = 0
            return p["uints"][0], p["uints"][1]
        self.last_error = 2
        return 0, -1

    def get_anno_layer_object_data(self) -> List[dict]:
        """Get metadata for all annotation objects."""
        msg_type, data = self._send_command(self.GETANNOLAYEROBJECTDATA)
        if msg_type != 1:
            self.last_error = msg_type
            return []

        arr = np.frombuffer(data, dtype=np.uint32)
        if len(arr) < 1:
            return []

        num = arr[0]
        objects, sp = [], 1
        for i in range(num):
            objects.append({
                "id": i + 1,
                "type": arr[sp] & 0xFFFF,
                "flags": (arr[sp] >> 16) & 0xFFFF,
                "col1": arr[sp + 1], "col2": arr[sp + 2],
                "anchorpoint": [arr[sp + 3], arr[sp + 4], arr[sp + 5]],
                "hierarchy": [arr[sp + 6], arr[sp + 7], arr[sp + 8], arr[sp + 9]],
                "collapsednr": arr[sp + 10],
                "boundingbox": [arr[sp + 11], arr[sp + 12], arr[sp + 13],
                               arr[sp + 14], arr[sp + 15], arr[sp + 16]],
            })
            sp += 21
        self.last_error = 0
        return objects

    def get_anno_layer_object_names(self) -> List[str]:
        """Get names of all annotation objects."""
        msg_type, data = self._send_command(self.GETANNOLAYEROBJECTNAMES)
        if msg_type != 1 or len(data) < 4:
            self.last_error = msg_type if msg_type != 1 else 2
            return []

        num = struct.unpack("<I", data[0:4])[0]
        names, pos = [], 4
        for _ in range(num):
            end = pos
            while end < len(data) and data[end] != 0:
                end += 1
            if end >= len(data):
                self.last_error = 2
                return []
            names.append(data[pos:end].decode("utf-8", errors="replace"))
            pos = end + 1
        self.last_error = 0
        return names

    def get_anno_object(self, object_id: int) -> dict:
        """Get complete annotation object data."""
        msg_type, data = self._send_command(self.GETANNOOBJECT, self._encode_uint32(object_id))
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        obj, success = self._decode_to_struct(data)
        if success != 1:
            self.last_error = 2
            return {}
        self.last_error = 0
        return obj

    def set_anno_object(self, object_id: int, obj_data: dict) -> int:
        """Update annotation object. Returns object ID on success, 0 on failure."""
        encoded, _ = self._encode_from_struct(obj_data)
        payload = self._encode_uint32(object_id) + encoded
        msg_type, data = self._send_command(self.SETANNOOBJECT, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def add_anno_object(self, ref_id: int, next_or_child: int, obj_data: dict) -> int:
        """Add new annotation object. Returns new ID or 0 on failure."""
        encoded, _ = self._encode_from_struct(obj_data)
        payload = self._encode_uint32(ref_id) + self._encode_uint32(next_or_child) + encoded
        msg_type, data = self._send_command(self.ADDANNOOBJECT, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def add_new_anno_object(self, ref_id: int, next_or_child: int, obj_type: int, name: str) -> int:
        """Add new annotation object (simple). Returns new ID or 0 on failure."""
        payload = (self._encode_uint32(ref_id) + self._encode_uint32(next_or_child) +
                   self._encode_uint32(obj_type) + self._encode_text(name))
        msg_type, data = self._send_command(self.ADDNEWANNOOBJECT, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def move_anno_object(self, obj_id: int, ref_id: int, next_or_child: int) -> bool:
        """Move annotation object in hierarchy."""
        payload = (self._encode_uint32(obj_id) + self._encode_uint32(ref_id) +
                   self._encode_uint32(next_or_child))
        msg_type, _ = self._send_command(self.MOVEANNOOBJECT, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def remove_anno_object(self, obj_id: int) -> bool:
        """Remove annotation object."""
        msg_type, _ = self._send_command(self.REMOVEANNOOBJECT, self._encode_uint32(obj_id))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_anno_object_nr(self) -> int:
        """Get selected annotation object. Returns -1 on failure."""
        msg_type, data = self._send_command(self.GETSELECTEDANNOOBJECTNR)
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return -1

    def set_selected_anno_object_nr(self, object_id: int) -> bool:
        """Select annotation object."""
        msg_type, _ = self._send_command(self.SETSELECTEDANNOOBJECTNR, self._encode_uint32(object_id))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    # ========================================================================
    # ANNOTATION NODE FUNCTIONS (SKELETONS)
    # ========================================================================

    def get_ao_node_data(self) -> Optional[np.ndarray]:
        """Get skeleton node data. Returns Nx14 array or None."""
        msg_type, data = self._send_command(self.GETAONODEDATA)
        if msg_type != 1:
            self.last_error = msg_type
            return None

        arr = np.frombuffer(data, dtype=np.uint32)
        if len(arr) < 1:
            self.last_error = 2
            return None

        num = arr[0]
        result = np.zeros((num, 14))
        sp = 1

        for i in range(num):
            result[i, 0] = i
            result[i, 1] = arr[sp] & 0xFF
            result[i, 2] = (arr[sp] >> 8) & 0xFF
            result[i, 3] = (arr[sp] >> 16) & 0xFF
            result[i, 4] = (arr[sp] >> 24) & 0xFF
            for j in range(6):
                v = arr[sp + 1 + j]
                result[i, 5 + j] = -1 if v == 0xFFFFFFFF else v
            result[i, 11] = np.frombuffer(data[4 * (sp + 7):4 * (sp + 9)], dtype=np.float64)[0]
            result[i, 12] = arr[sp + 9]
            result[i, 13] = arr[sp + 10]
            sp += 11

        self.last_error = 0
        return result

    def get_ao_node_labels(self) -> Tuple[List[int], List[str]]:
        """Get skeleton node labels."""
        msg_type, data = self._send_command(self.GETAONODELABELS)
        if msg_type != 1 or len(data) < 4:
            self.last_error = msg_type if msg_type != 1 else 2
            return [], []

        num = struct.unpack("<I", data[0:4])[0]
        if num == 0:
            self.last_error = 0
            return [], []

        nodes = list(np.frombuffer(data[4:4 + num * 4], dtype=np.uint32))
        labels, pos = [], 4 + num * 4

        for _ in range(num):
            end = pos
            while end < len(data) and data[end] != 0:
                end += 1
            if end >= len(data):
                self.last_error = 2
                return [], []
            labels.append(data[pos:end].decode("utf-8", errors="replace"))
            pos = end + 1

        self.last_error = 0
        return nodes, labels

    def get_ao_node_params(self, anno_object_id: int, node_dfsnr: int) -> dict:
        """Get detailed node parameters."""
        payload = self._encode_uint32(anno_object_id) + self._encode_uint32(node_dfsnr)
        msg_type, data = self._send_command(self.GETAONODEPARAMS, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return {}
        obj, success = self._decode_to_struct(data)
        if success != 1:
            self.last_error = 2
            return {}
        self.last_error = 0
        return obj

    def set_ao_node_params(self, anno_object_id: int, node_dfsnr: int, params: dict) -> bool:
        """Set node parameters."""
        encoded, _ = self._encode_from_struct(params)
        payload = self._encode_uint32(anno_object_id) + self._encode_uint32(node_dfsnr) + encoded
        msg_type, _ = self._send_command(self.SETAONODEPARAMS, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_selected_ao_node_by_dfsnr(self, node_dfsnr: int) -> bool:
        """Select node by DFS number."""
        msg_type, _ = self._send_command(self.SETSELECTEDAONODEBYDFSNR, self._encode_uint32(node_dfsnr))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_selected_ao_node_by_coords(self, x: int, y: int, z: int) -> bool:
        """Select node closest to coordinates."""
        payload = self._encode_uint32(x) + self._encode_uint32(y) + self._encode_uint32(z)
        msg_type, _ = self._send_command(self.SETSELECTEDAONODEBYCOORDS, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_selected_ao_node_nr(self) -> int:
        """Get selected node DFS number. Returns -1 on failure."""
        msg_type, data = self._send_command(self.GETSELECTEDAONODENR)
        if msg_type != 1:
            self.last_error = msg_type
            return -1
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return -1

    def add_ao_node(self, x: int, y: int, z: int) -> bool:
        """Add node to selected skeleton."""
        payload = self._encode_uint32(x) + self._encode_uint32(y) + self._encode_uint32(z)
        msg_type, _ = self._send_command(self.ADDAONODE, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def move_selected_ao_node(self, x: int, y: int, z: int) -> bool:
        """Move selected node to new coordinates."""
        payload = self._encode_uint32(x) + self._encode_uint32(y) + self._encode_uint32(z)
        msg_type, _ = self._send_command(self.MOVESELECTEDAONODE, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def remove_selected_ao_node(self) -> bool:
        """Remove selected node."""
        msg_type, _ = self._send_command(self.REMOVESELECTEDAONODE)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def swap_selected_ao_node_children(self) -> bool:
        """Swap children of selected node."""
        msg_type, _ = self._send_command(self.SWAPSELECTEDAONODECHILDREN)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def make_selected_ao_node_root(self) -> bool:
        """Make selected node the skeleton root."""
        msg_type, _ = self._send_command(self.MAKESELECTEDAONODEROOT)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def split_selected_skeleton(self, new_root_dfsnr: int, new_name: str) -> int:
        """Split skeleton at node. Returns new skeleton ID or 0."""
        payload = self._encode_uint32(new_root_dfsnr) + self._encode_text(new_name)
        msg_type, data = self._send_command(self.SPLITSELECTEDSKELETON, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return 0
        p = self._parse_payload(data)
        if len(p["uints"]) == 1:
            self.last_error = 0
            return p["uints"][0]
        self.last_error = 2
        return 0

    def weld_skeletons(self, obj1: int, node1: int, obj2: int, node2: int) -> bool:
        """Weld two skeletons at specified nodes."""
        payload = (self._encode_uint32(obj1) + self._encode_uint32(node1) +
                   self._encode_uint32(obj2) + self._encode_uint32(node2))
        msg_type, _ = self._send_command(self.WELDSKELETONS, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def get_closest_ao_node_by_coords(self, x: int, y: int, z: int, max_dist: int) -> Tuple[int, int, float]:
        """Find closest node. Returns (object_id, node_dfsnr, distance) or (-1, -1, -1)."""
        payload = (self._encode_uint32(x) + self._encode_uint32(y) +
                   self._encode_uint32(z) + self._encode_uint32(max_dist))
        msg_type, data = self._send_command(self.GETCLOSESTAONODEBYCOORDS, payload)
        if msg_type != 1:
            self.last_error = msg_type
            return -1, -1, -1
        p = self._parse_payload(data)
        if len(p["ints"]) == 2 and len(p["doubles"]) == 1:
            self.last_error = 0
            return p["ints"][0], p["ints"][1], p["doubles"][0]
        self.last_error = 0
        return -1, -1, -1

    # ========================================================================
    # EXECUTE FUNCTIONS
    # ========================================================================

    def execute_fill(self, source_layer: int, target_layer: int,
                     x: int, y: int, z: int, mip: int) -> bool:
        """Execute fill operation."""
        payload = (self._encode_uint32(source_layer) + self._encode_uint32(target_layer) +
                   self._encode_uint32(x) + self._encode_uint32(y) +
                   self._encode_uint32(z) + self._encode_uint32(mip))
        msg_type, _ = self._send_command(self.EXECUTEFILL, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def execute_limited_fill(self, source_layer: int, target_layer: int,
                              x: int, y: int, z: int, mip: int,
                              minx: int, maxx: int, miny: int, maxy: int,
                              minz: int, maxz: int) -> bool:
        """Execute limited fill operation within bounds."""
        payload = (self._encode_uint32(source_layer) + self._encode_uint32(target_layer) +
                   self._encode_uint32(x) + self._encode_uint32(y) +
                   self._encode_uint32(z) + self._encode_uint32(mip) +
                   self._encode_uint32(minx) + self._encode_uint32(maxx) +
                   self._encode_uint32(miny) + self._encode_uint32(maxy) +
                   self._encode_uint32(minz) + self._encode_uint32(maxz))
        msg_type, _ = self._send_command(self.EXECUTELIMITEDFILL, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    # ========================================================================
    # UI CONTROL
    # ========================================================================

    def set_ui_mode(self, mode: int) -> bool:
        """Set UI mode."""
        msg_type, _ = self._send_command(self.SETUIMODE, self._encode_uint32(mode))
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_2d_view_orientation(self, orientation: int) -> bool:
        """Set 2D view orientation."""
        payload = self._encode_int32(orientation) + self._encode_int32(0)
        msg_type, _ = self._send_command(self.SET2DVIEWORIENTATION, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_error_popups_enabled(self, code: int, enabled: bool) -> bool:
        """Enable/disable error popups for specific code."""
        payload = self._encode_uint32(code) + self._encode_uint32(int(enabled))
        msg_type, _ = self._send_command(self.SETERRORPOPUPSENABLED, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success

    def set_popups_enabled(self, enabled: bool) -> bool:
        """Enable/disable all popups."""
        payload = self._encode_uint32(0) + self._encode_uint32(int(enabled))
        msg_type, _ = self._send_command(self.SETPOPUPSENABLED, payload)
        success = msg_type == 1
        self.last_error = 0 if success else msg_type
        return success
