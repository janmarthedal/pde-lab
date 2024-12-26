import argparse


def gen_line(x1, x2, meshsize, out):
    import gmsh

    gmsh.initialize()
    point_a = gmsh.model.geo.addPoint(x1, 0.0, 0.0, meshSize=meshsize)
    point_b = gmsh.model.geo.addPoint(x2, 0.0, 0.0, meshSize=meshsize)
    gmsh.model.geo.addLine(point_a, point_b)  # ignore line id tag
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.write(out)
    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gen-mesh.py", description="Generates test meshes"
    )
    parser.add_argument(
        "--out", type=str, metavar="filename", default="mesh.msh", help="Output file"
    )
    subparsers = parser.add_subparsers(
        title="Geometry", dest="geometry", help="subcommand help", required=True
    )

    # create the parser for the "line" command
    parser_a = subparsers.add_parser("line", help="Create line mesh")
    parser_a.add_argument("x1", type=float, help="Starting x coordinate")
    parser_a.add_argument("x2", type=float, help="Ending x coordinate")
    parser_a.add_argument("meshsize", type=float, help="Mesh size")

    # create the parser for the "rect" command
    # parser_b = subparsers.add_parser('rect', help='Create rectangle help')
    # parser_b.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')

    args = parser.parse_args()
    if args.geometry == "line":
        gen_line(args.x1, args.x2, args.meshsize, args.out)
