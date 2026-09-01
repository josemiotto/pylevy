"""``levy-tables``: regenerate the lookup tables.

    levy-tables build                    # into the user cache directory
    levy-tables build --out ./tables     # somewhere explicit
    levy-tables build --size 40,16,21 --jobs 6
    levy-tables where                    # which tables are actually in use

Previously the only way to do this was ``python -m levy build``, which wrote
24 MB straight into the installed package -- impossible on a read-only or
system install, and silently destructive on a partial run.
"""

import argparse
import logging
import os
import sys
import time

logger = logging.getLogger("levy._build")


def _parse_size(text):
    """Parse an ``x,alpha,beta`` grid size from the command line.

    Parameters
    ----------
    text : str
        Three comma-separated integers, e.g. ``"200,76,101"``.

    Returns
    -------
    tuple of int
        The grid shape.

    Raises
    ------
    argparse.ArgumentTypeError
        If there are not three of them, or any is below 8, which is the
        smallest grid cubic interpolation can use.
    """
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected three comma-separated integers, e.g. 200,76,101")
    if any(p < 8 for p in parts):
        raise argparse.ArgumentTypeError(
            "each dimension must be at least 8 for cubic interpolation")
    return tuple(parts)


def _parse_what(text):
    """Parse the comma-separated list of tables to build.

    Parameters
    ----------
    text : str
        Any of ``pdf``, ``cdf`` and ``limits``, comma-separated.

    Returns
    -------
    list of str
        The requested tables, in the order given.

    Raises
    ------
    argparse.ArgumentTypeError
        If any name is not one of the three.
    """
    allowed = {"pdf", "cdf", "limits"}
    what = [p.strip() for p in text.split(",") if p.strip()]
    unknown = set(what) - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown table(s): {}".format(", ".join(sorted(unknown))))
    return what


def build(args):
    """Run the ``build`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments: ``out``, ``size``, ``what`` and ``jobs``.

    Returns
    -------
    int
        Process exit status.
    """
    from levy import data_dir
    from levy._build.tables import build_crossover_tables, build_density_tables, write_manifest

    out_dir = args.out or data_dir(writable=True)
    logger.info("Writing tables to %s", out_dir)

    started = time.time()
    densities = [w for w in args.what if w in ("pdf", "cdf")]
    cdf_table = None
    if densities:
        results = build_density_tables(out_dir, args.size, jobs=args.jobs, what=densities)
        if "cdf" in results:
            cdf_table = results["cdf"][0]

    if "limits" in args.what:
        if cdf_table is None and args.size != tuple(__import__("levy").size):
            logger.error(
                "--what limits at a non-default --size needs the cdf table from the same run; "
                "add cdf to --what"
            )
            return 2
        build_crossover_tables(out_dir, args.size, jobs=args.jobs, cdf_table=cdf_table)

    manifest = write_manifest(
        out_dir, args.size, extra={"seconds": round(time.time() - started, 1)})
    logger.info("Done in %.1fs. Manifest: %s",
                time.time() - started, os.path.join(out_dir, "manifest.json"))
    for name, entry in sorted(manifest["tables"].items()):
        logger.info("  %-12s %8.2f MB  %s", name, entry["bytes"] / 1e6, entry["sha256"][:16])
    return 0


def where(args):
    """Run the ``where`` subcommand, reporting which tables are in use.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Unused; present for the subcommand dispatch.

    Returns
    -------
    int
        Process exit status.
    """
    import levy

    print(f"tables in use : {levy.data_dir()}")
    print(f"packaged      : {levy.PACKAGED_DATA}")
    print(f"user cache    : {levy.user_cache_dir()}")
    print("LEVY_DATA_DIR : {}".format(os.environ.get("LEVY_DATA_DIR", "(unset)")))
    directory = levy.data_dir()
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if not name.endswith((".npz", ".json")):
            continue
        size_mb = os.path.getsize(os.path.join(directory, name)) / 1e6
        print(f"  {name:<16} {size_mb:8.2f} MB")
    return 0


def main(argv=None):
    """Entry point for the ``levy-tables`` console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Read from ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(prog="levy-tables", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose (DEBUG) logging")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="regenerate the lookup tables")
    build_parser.add_argument(
        "--out", help="output directory (default: the user cache directory)")
    build_parser.add_argument("--size", type=_parse_size, default=None,
                              help="grid as x,alpha,beta (default: 200,76,101)")
    build_parser.add_argument("--what", type=_parse_what, default=["pdf", "cdf", "limits"],
                              help="which tables to build (default: pdf,cdf,limits)")
    build_parser.add_argument("--jobs", type=int, default=1,
                              help="worker processes; a full build is ~55 CPU-minutes")
    build_parser.set_defaults(func=build)

    where_parser = subparsers.add_parser("where", help="show which tables are in use")
    where_parser.set_defaults(func=where)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.command == "build" and args.size is None:
        import levy
        args.size = tuple(levy.size)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
