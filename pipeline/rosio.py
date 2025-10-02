from typing import Iterable, Tuple
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


def open_depth_bag(bag_dir: str = "depth", topic: str = "/depth") -> Tuple[Reader, list]:
    reader = Reader(bag_dir)
    connections = []
    reader.__enter__()
    try:
        connections = [x for x in reader.connections if x.topic == topic]
    except Exception:
        reader.__exit__(None, None, None)
        raise
    return reader, connections


def iterate_depth_frames(reader: Reader, connections) -> Iterable[Tuple[int, int, np.ndarray]]:
    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = deserialize_cdr(rawdata, conn.msgtype)
        h, w = msg.height, msg.width
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32)
        if depth.max() > 1000:
            depth = depth / 1000.0
        yield h, w, timestamp, depth


def close_reader(reader: Reader) -> None:
    reader.__exit__(None, None, None)


