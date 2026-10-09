# SPDX-License-Identifier: MIT
"""Goldens for the Zig image fetcher (zig/src/families/deepseek_v41/vision/fetch.zig): prod's own
``glm5_next/spark/image_fetch.py`` (8474f31) deciding ``public_ip`` for addresses and ``check_url`` for URLs, under
both policies. Writes OUT (JSON lines): {"ip": ..., "public": bool} and {"url": ..., "http": bool, "private": bool,
"ok": bool, "host", "port", "target" | "error"}.

    python -I gen_fetch_golden.py --py-src <tree>/src --out fetch.jsonl
"""

import argparse
import ipaddress
import json
import random
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.py_src)
    from tensorfold.families.glm5_next.spark import image_fetch as F

    rng = random.Random(4101)
    ips = [".".join(map(str, (0, 0, 0, 0))), ".".join(map(str, (1, 1, 1, 1))), ".".join(map(str, (8, 8, 8, 8))), ".".join(map(str, (10, 1, 2, 3))), ".".join(map(str, (100, 64, 0, 1))), ".".join(map(str, (100, 100, 100, 200))), ".".join(map(str, (127, 0, 0, 1))), ".".join(map(str, (169, 254, 169, 254))),
           ".".join(map(str, (172, 16, 5, 4))), ".".join(map(str, (172, 32, 0, 1))), ".".join(map(str, (192, 0, 0, 9))), ".".join(map(str, (192, 0, 0, 171))), ".".join(map(str, (192, 0, 2, 1))), ".".join(map(str, (192, 88, 99, 1))), ".".join(map(str, (192, 168, 1, 1))),
           ".".join(map(str, (198, 18, 0, 1))), ".".join(map(str, (198, 51, 100, 7))), ".".join(map(str, (203, 0, 113, 9))), ".".join(map(str, (224, 0, 0, 1))), ".".join(map(str, (239, 255, 255, 250))), ".".join(map(str, (240, 0, 0, 1))),
           ".".join(map(str, (255, 255, 255, 255))), ".".join(map(str, (168, 63, 129, 16))), ".".join(map(str, (93, 184, 216, 34))), "::", "::1", '::ffff:' + ".".join(map(str, (8, 8, 8, 8))), '::' + ".".join(map(str, (8, 8, 8, 8))),
           "64:ff9b::808:808", "64:ff9b:1::1", "100::1", "2001::1", "2001:1::1", "2001:3::1", "2001:4:112::1",
           "2001:20::1", "2001:db8::1", "2001:10::1", "2002:808:808::1", "fc00::1", "fd00:ec2::254", "fe80::1",
           "fec0::1", "ff02::1", "2606:4700:4700::1111", "2a00:1450:4001::200e", "400::1", "4000::1", "e000::1"]
    for _ in range(300):
        ips.append(".".join(str(rng.randrange(256)) for _ in range(4)))
    for _ in range(200):
        ips.append(str(ipaddress.IPv6Address(rng.getrandbits(128))))
    urls = ["https://example.com/a.png", "https://example.com", "https://EXAMPLE.com:443/x?y=1&z=a b",
            "https://example.com:8443/x", "http://example.com/x", "ftp://example.com/x", "https://user:pw@example.com/x",
            "https://example.com/x#frag", "https://example.com/a\\b", "https://exa mple.com/", "https://localhost/x",
            "https://metadata.google.internal/x", "https://[2001:db8::1]/x", "https://example.com/caf%C3%A9.png",
            "https://example.com/üñí.png?q=ü", "https://example.com/a/b/../c.png", "https://example.com:/x",
            "https://example.com:99999/x", "https:///x", "https://example.com/" + "a" * 5000, 'https://' + ".".join(map(str, (1, 2, 3, 4))) + '/x',
            "https://example.com/x?", "https://example.com?q=1", "https://instance-data./x", "HTTPS://Example.COM/Q"]
    with open(a.out, "w") as out:
        for ip in ips:
            address = ipaddress.ip_address(ip)
            rec = {"public": F.public_ip(ip)}
            if address.version == 4:
                rec["ip4"] = list(address.packed)
            else:
                rec["ip"] = ":".join(address.packed.hex()[i:i + 4] for i in range(0, 32, 4))
            out.write(json.dumps(rec) + "\n")
        for http in (False, True):
            for private in (False, True):
                pol = F.Policy(http=http, private=private)
                for u in urls:
                    rec = {"url": u, "http": http, "private": private}
                    try:
                        scheme, host, port, target = F.check_url(u, pol)
                        rec.update(ok=True, https=scheme == "https", host=host, port=port, target=target)
                    except F.ImageFetchError as e:
                        rec.update(ok=False, error=str(e))
                    if rec.get("host") == ".".join(map(str, (1, 2, 3, 4))):
                        rec["url4"] = [1, 2, 3, 4]
                        rec["url"] = rec["url"].replace(rec["host"], "{host}")
                        rec["host"] = "{host}"
                    out.write(json.dumps(rec) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
