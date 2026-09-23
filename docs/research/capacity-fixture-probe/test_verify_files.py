import struct
import unittest
from verify_files import parse_graph


def golden():
    # Independently constructed two-node graph, one upper layer; distinct fields.
    header = b"VNDB" + struct.pack("<III8QiIQ", 2, 1, 0, 2, 2, 4, 5, 16, 32, 42, 0, 1, 1, 0)
    node0 = struct.pack("<QII2fQQQ", 101, 1, 0, 1., 0., 1, 1, 0)
    node1 = struct.pack("<QII2fQQ", 202, 0, 0, 0., 1., 1, 0)
    return header + node0 + node1


class ParserChecks(unittest.TestCase):
    def test_golden(self):
        data = golden()
        got = parse_graph(data)
        self.assertEqual(got["nodes_by_level"], [1, 1])
        self.assertEqual(got["total_links"], 2)
        self.assertEqual(got["exact_fit_neighbor_bytes_64bit"], 136)
        self.assertEqual(len(data), 184)

    def test_every_truncation(self):
        data = golden()
        for end in range(len(data)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                parse_graph(data[:end])

    def test_corruptions(self):
        for offset, fmt, value in [(4, "I", 3), (8, "I", 0), (12, "I", 3),
                                    (16, "Q", 0), (24, "Q", 1000000), (32, "Q", 1),
                                    (40, "Q", 1), (48, "Q", 0), (72, "Q", 2),
                                    (80, "i", 0), (84, "I", 2), (88, "Q", 1),
                                    (104, "I", 33), (108, "I", 2),
                                    (120, "Q", 2), (128, "Q", 0), (128, "Q", 2)]:
            bad = bytearray(golden())
            struct.pack_into("<" + fmt, bad, offset, value)
            with self.subTest(offset=offset, value=value), self.assertRaises(ValueError):
                parse_graph(bad)
        with self.assertRaises(ValueError):
            parse_graph(golden() + b"trailing")


if __name__ == "__main__":
    unittest.main()
