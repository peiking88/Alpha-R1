"""Universe (instrument list) loaders."""

from pathlib import Path


def load_zxg(path: str = "/home/li/.local/share/tdxcfv/drive_c/tc/T0002/blocknew/zxg.blk",
             ignore_hk: bool = True) -> list[str]:
    """Parse a 通达信 zxg.blk file into ``[SH600000, SZ000001, ...]``.

    Format: one code per line.  Prefix ``1`` = Shanghai, ``0`` = Shenzhen,
    ``31#`` = Shanghai ETF/fund.  Each line is ``<prefix><6-digit-code>``.

    Args:
        ignore_hk: skip Hong Kong stocks (prefix ``11`` or ``00``-prefixed
            5-digit codes that don't match A-share patterns).
    """
    instruments = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("31#"):
            # ETF/fund codes like 31#01879 → SH01879
            instruments.append("SH" + line[3:])
        elif line.startswith("1"):
            code = line[1:]
            if ignore_hk and code.startswith("1"):
                continue  # 11xxxx = HK stock
            instruments.append("SH" + code)
        elif line.startswith("0"):
            code = line[1:]
            if ignore_hk and len(code) == 5:
                continue  # 0 + 5-digit = HK stock
            instruments.append("SZ" + code)
    return instruments
