#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nibabel as nib
from trimeshpy import TriMesh_Vtk
import numpy as np
import os
import csv
from dipy.viz import window, actor
import dipy.io.vtk as io_vtk
from dipy.tracking.streamline import set_number_of_points
from scipy import interpolate, spatial
import vtk
from scilpy.utils.streamlines import load_in_voxel_space
from vtk.util import numpy_support
from gg_code.utils.bounding_rectangle import min_bounding_rect
import gg_code.utils.compute_surface as cs
from gg_code.utils.chanvese3d import chanvese3d
import gg_code.utils.rotating_calipers as rc


def compute_intersection_with_plane(cen, vec_tangent, vtk_mesh):
    # compute intersection
    plane = vtk.vtkPlane()
    plane.SetOrigin(cen[0], cen[1], cen[2])
    plane.SetNormal(vec_tangent[0], vec_tangent[1], vec_tangent[2])
    planeCut = vtk.vtkCutter()
    planeCut.SetInputData(vtk_mesh)
    planeCut.SetCutFunction(plane)
    planeCut.Update()
    intersection = planeCut.GetOutput()  # type: object

    return intersection


def update_normals(vtk_mesh):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(vtk_mesh)
    normals.SetComputeCellNormals(True)
    normals.SetComputePointNormals(True)
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def tangent(cen):
    vector_tan = np.empty(cen.shape)
    for i in xrange(cen.shape[0]):
        if i == 29:
            vector_tan[i] = vector_tan[i-1]
        else:
            dist = np.sqrt((cen[i][0]-cen[i+1][0])**2+(cen[i][1]-cen[i+1][1])**2+(cen[i][2]-cen[i+1][2])**2)
            vector_tan[i] = (cen[i+1] - cen [i])/dist
    return vector_tan


def vtk2np(vtk_points):
    vtk_points = vtk_points.GetPoints().GetData()

    pts = numpy_support.vtk_to_numpy(vtk_points)
    return pts


def curve2mesh(curve):
    if curve == 0:
        return 0
    strips = vtk.vtkStripper()
    strips.SetInputData(curve)
    strips.Update()

    int_poly = vtk.vtkPolyData()
    int_poly.SetPoints(strips.GetOutput().GetPoints())
    int_poly.SetPolys(strips.GetOutput().GetLines())
    mesh = vtk.vtkTriangleFilter()
    mesh.SetInputData(int_poly)
    mesh.Update()

    return mesh


def curve_intersections(curve1, curve2):
    mesh1 = curve2mesh(curve1)
    mesh2 = curve2mesh(curve2)
    int_sctPF = vtk.vtkIntersectionPolyDataFilter()
    int_sctPF.SetInputData(0, mesh1.GetOutput())
    int_sctPF.SetInputData(1, mesh2.GetOutput())
    int_sctPF.Update()
    intersection = int_sctPF.GetOutput()
    nb_lines = intersection.GetNumberOfLines()
    return nb_lines


def slice_xyplane(points, plane_normal):
    n = np.array([0, 0, 1])
    u = np.cross(plane_normal, n)
    c_theta = np.dot(n, plane_normal)
    s_theta = np.sin(np.arccos(np.dot(n, plane_normal)))

    Rot = np.array([[u[0]**2+(1-u[0]**2)*c_theta, u[0]*u[1]*(1-c_theta)-u[2]*s_theta, u[0]*u[2]*(1-c_theta)-u[1]*s_theta],
                    [u[0]*u[1]*(1-c_theta)-u[2]*s_theta, u[1]**2+(1-u[1]**2)*c_theta, u[0]*u[2]*(1-c_theta)-u[1]*s_theta],
                    [u[0]*u[2]*(1-c_theta)-u[1]*s_theta, u[0]*u[2]*(1-c_theta)-u[1]*s_theta, u[2]**2+(1-u[2]**2)*c_theta]])

    points = points.T
    points = np.dot(Rot, points).T
    points = np.delete(points, (2, 3), 1)

    return points


def compute_perimeter_vtk_shape(shape):
    vtk_points = shape.GetPoints().GetData()
    pts = numpy_support.vtk_to_numpy(vtk_points)
    strips = vtk.vtkStripper()
    strips.SetInputData(shape)
    strips.Update()
    shape_poly = vtk.vtkPolyData()
    shape_poly.SetPoints(strips.GetOutput().GetPoints())
    shape_poly.SetPolys(strips.GetOutput().GetLines())
    nb_polys = shape_poly.GetNumberOfPolys()
    lines = numpy_support.vtk_to_numpy(strips.GetOutput().GetLines().GetData())
    per = 0
    # if nb_polys > 1:
    #     print("WARNING: The intersection is formed of two polygons or more. The metrics may be difficult to interpret.")

    i = 1
    for j in xrange(nb_polys):
        nb_pts = i + lines[i-1] - 1
        while i != nb_pts:
            per = per + spatial.distance.euclidean(pts[lines[i]], pts[lines[i+1]])
            i += 1
        i += 2
    return per


def compute_convexHull(shape, vec_tangent):
    vtk_points = shape.GetPoints().GetData()

    pts_3D = numpy_support.vtk_to_numpy(vtk_points)
    pts = slice_xyplane(pts_3D, vec_tangent)

    hull = spatial.ConvexHull(pts)
    return hull, pts


def compute_area_intersection(int_sct):
    int_mesh = curve2mesh(int_sct)

    mesh_info = vtk.vtkMassProperties()
    mesh_info.SetInputData(int_mesh.GetOutput())
    mesh_info.Update()
    area = mesh_info.GetSurfaceArea()

    return area


def compute_ratio_bottleneck(area_max, area_min):
    area_bottleneck = area_min
    return area_max/area_bottleneck


def compute_elongation(shape, vec_tangent):
    hull, pts = compute_convexHull(shape, vec_tangent)

    hull_pts = np.empty((hull.vertices.shape[0]+1, 2))
    i = 0

    for vrt in hull.vertices:
        hull_pts[i] = pts[vrt]
        i += 1
    
    hull_pts[i] = pts[hull.vertices[0]]

    info = min_bounding_rect(hull_pts)
    width = min(info[2], info[3])
    height = max(info[2], info[3])
    return height/width


def compute_irregularity(shape, area):
    perimeter = compute_perimeter_vtk_shape(shape)
    irr = (perimeter**2) / (4*np.pi*area)
    return irr


def compute_paris_factor(shape, vec_tangent):
    per_shape = compute_perimeter_vtk_shape(shape)
 
    hull, pts = compute_convexHull(shape, vec_tangent)

    per_cH = 0
    vertices = hull.vertices
    for i in xrange(vertices.shape[0]-1):
        per_cH = per_cH + spatial.distance.euclidean(pts[vertices[i]], pts[vertices[i+1]])
    
    p = vertices.shape[0]
    per_cH = per_cH + spatial.distance.euclidean(pts[vertices[vertices.shape[0]-1]], pts[vertices[0]])

    paris_factor = 2*((per_shape-per_cH)/per_cH) * 100

    return paris_factor


def compute_delta_factor(shape, area, vec_tangent):
    hull, pts = compute_convexHull(shape, vec_tangent)
    area_hull = hull.area
    delta_factor = ((area_hull-area)/area) * 100
    return delta_factor


def compute_aspect_ratio(shape, vec_tangent):
    points = vtk2np(shape)
    points_xy = slice_xyplane(points, vec_tangent)
    min = rc.min_width(points_xy)
    max = rc.max_width(points_xy)
    return max/min


def save_intersection(input, filename):
    #save file in vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(input)
    else:
        writer.SetInputData(input)
    writer.Update()
    writer.Write()


def visu_intersection(input, linewidth=1, color=[0,1,0]):
    slice_actor = ut_vtk.get_actor_from_polydata(input)
    slice_actor.GetProperty().SetColor(*color)
    slice_actor.GetProperty().SetLineWidth(linewidth)
    return slice_actor


def visu_mesh(mesh, name):
    my_vertices = mesh.get_vertices()
    my_triangles = mesh.get_triangles()
    my_polydata = vtk.vtkPolyData()

    ut_vtk.set_polydata_vertices(my_polydata, my_vertices)
    ut_vtk.set_polydata_triangles(my_polydata, my_triangles)

    my_polydata = update_normals(my_polydata)

    file_name = "/home/local/USHERBROOKE/greg2707/Recherche/gg_code/scripts/Fanning/Data/Left_vs_right/Surface/" \
                + name + "_surface.vtk"
    io_vtk.save_polydata(my_polydata, file_name)

    # get vtkActor
    surf_actor = ut_vtk.get_actor_from_polydata(my_polydata)
    surf_actor.GetProperty().SetOpacity(0.4)

    # renderer and scene
    return surf_actor


def visu_centroid(centroid):
    return actor.line(centroid, linewidth=3)


def visu_dots(dots, color):
    return actor.dots(dots, color)


if __name__ == "__main__":
    # data path
    data_path = "/home/local/USHERBROOKE/greg2707/Recherche/Data/PentheraT1.5/Results/Final_tracks/"
    save_path = "/home/local/USHERBROOKE/greg2707/Recherche/gg_code/scripts/Fanning/Data"

    # n = "_rpt_m_warp_ic"
    
    bundle_ids = ['Audrey_Hardi', 'Audrey_Hardi_2', 'Audrey_Hardi_3', 'David_Provencher', 'David_Provencher_2', 'David_Provencher_3',
                 'Emmanuelle_Renauld', 'Emmanuelle_Renauld_2', 'Emmanuelle_Renauld_3', 'Francois_Trottier', 'Francois_Trottier_2', 'Francois_Trottier_3',
                 'Marc_Cote', 'Marc_Cote_2', 'Marc_Cote_3', 'Marianne_Rheault', 'Marianne_Rheault_2', 'Marianne_Rheault_3',
                 'Maxime_Chamberland', 'Maxime_Chamberland_2', 'Maxime_Chamberland_3', 'Michael_Bernier', 'Michael_Bernier_2', 'Michael_Bernier_3',
                 'Stephanie_Madrolle', 'Stephanie_Madrolle_2', 'Stephanie_Madrolle_3', 'Vincent_Methot', 'Vincent_Methot_2', 'Vincent_Methot_3']
    for j in xrange(len(bundle_ids)):
        input_anat = nib.load(data_path + "template0.nii.gz")
        trk_file = data_path + "PT_left/" + bundle_ids[j] + "_PT_L_template_wl_n_ic.trk"
        input_trk = nib.streamlines.load(trk_file)
        binary_mask_file = save_path + "/binary_mask.nii.gz"
        volume_file = save_path + "/volume.nii.gz"
        surface_file = save_path + "/pre_surface.vtk"
        centroid_file = data_path + "Centroid/" + bundle_ids[j] + "_PT_L_centroid.trk"

        # Generate and save binary mask of the track
        binary_mask = cs.binary_mask(input_trk, input_anat)
        img = nib.Nifti1Image(binary_mask, input_anat.affine, input_anat.header)
        nib.save(img, binary_mask_file)

        # Compute phi function with the chanvese algo
        ds = 1
        nib_file = nib.load(binary_mask_file)
        img = nib_file.get_data()[::ds, ::ds, ::ds]
        mask = nib.load(binary_mask_file).get_data()[::ds, ::ds, ::ds]
        affine = nib_file.get_affine()
        phi = chanvese3d(img, mask, max_its=4, alpha=1, display=True)

        # From the phi function, compute with the marching cube the surface
        volume = cs.surf_from_phi(phi)
        img = nib.Nifti1Image(volume, input_anat.affine, input_anat.header)
        nib.save(img, volume_file)
        os.system(
            "se_vol2surf.py " + volume_file + " --out_surface " + surface_file + " --max_label")

        # Smoothing of the surface and adjust normals
        mesh = cs.laplacian_smoothing(surface_file)
        mesh.update_normals()
        normals = mesh.get_normals()
        normals = normals * (-1)
        mesh.set_normals(normals)
        vtk_mesh = mesh.get_polydata(update_normal=True)
        vtk_mesh = update_normals(vtk_mesh)


        #inverse if necessary the centroid
        centroid = load_in_voxel_space(centroid_file, input_anat)
        cen = np.squeeze(centroid)
        if cen[:,2][0] > 75:
            cen = cen[::-1]

        # renderer = window.Renderer()
        area = np.empty(cen.shape[0])
        vec_tangent = tangent(cen)
        int_sct = []
        csv_data = []

        for i in xrange(cen.shape[0]):
            int_sct.append(compute_intersection_with_plane(cen[i], vec_tangent[i], vtk_mesh))
            area[i] = compute_area_intersection(int_sct[i])


        id_area_max = area.argmax()
        id_area_min = area.argmin()
        area_max = area.max()
        area_min = area.min()
        csv_data = []

        for i in xrange(cen.shape[0]):
            dt = [bundle_ids[j], i, area[i], compute_ratio_bottleneck(area[i], area_min), compute_elongation(int_sct[i], vec_tangent[i]), 
                compute_irregularity(int_sct[i], area[i]), compute_paris_factor(int_sct[i], vec_tangent[i]), compute_delta_factor(int_sct[i], area[i], vec_tangent[i]),
                compute_aspect_ratio(int_sct[i], vec_tangent[i])]
            csv_data.append(dt)
        #     if i == 20 or i == 21 or i == 22:
        #         save_intersection(int_sct[i], save_path + "/Intersections/" + bundle_ids[j] + "_" + str(i) + "_right_int.vtk" )
        #     # if i in [id_area_max]:
        #         # renderer.add(visu_intersection(int_sct[i], 6, [1, 0, 0]))
        #     # else:
        #     # renderer.add(visu_intersection(int_sct[i], linewidth=2))

        # with open(save_path + "/Left_vs_right/Results_left_new_measures.csv", 'a') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerows(csv_data)
        # csv_file.close()

        # visu
        # renderer.add(visu_dots(cen, [0,1,1]))
        # # renderer.add(visu_dots((vec_tangent + cen), [1,0,1]))
        # renderer.add(visu_mesh(mesh, "rpt_" + bundle_ids[j]))
        # renderer.add(visu_centroid(centroid))
        # renderer.set_camera(position=[-38.97, -177.38, -3.02],
        #            view_up=[-0.15, -0.18, 0.97],
        #            focal_point=[84.29, 91.31, 66.65])
        # window.show(renderer, size=(800, 1080), reset_camera=True)
        # renderer.camera_info()
        # window.snapshot(renderer, size=(800, 1080), offscreen=True, 
        #                 fname="/home/local/USHERBROOKE/greg2707/Recherche/Presentation/GIF/29.png")

