import gmsh

delx = 0.5
xend = 20

gmsh.initialize()

point_a = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, meshSize=delx)
point_b = gmsh.model.geo.addPoint(xend, 0.0, 0.0, meshSize=delx)

line = gmsh.model.geo.addLine(point_a, point_b)

gmsh.model.geo.synchronize()

# make a 1d mesh
gmsh.model.mesh.generate(1)

# save the mesh as a NASTRAN bulk data file
gmsh.write("line.msh")

gmsh.finalize()
