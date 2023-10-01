import pickle

import astropy.table as at
import numpy as np
from galpy.util import coords
from scipy import stats as stats


def rotation_axis_angle(axis):
    # function rotation_axis_angle, axis
    unitz = [0.0, 0.0, 1.0]
    angle = np.pi + (-1.0) * np.arccos(
        axis[2] / np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    )
    u = np.zeros(3)
    u[0] = -unitz[2] * axis[1]
    u[1] = unitz[2] * axis[0]
    u[2] = 0.0
    n = np.sqrt(u[0] ** 2 + u[1] ** 2)
    u = u / n
    kronecker = np.zeros(shape=(3, 3))
    kronecker[0, 0] = 1.0
    kronecker[0, 1] = 0.0
    kronecker[0, 2] = 0.0
    kronecker[1, 0] = 0.0
    kronecker[1, 1] = 1.0
    kronecker[1, 2] = 0.0
    kronecker[2, 0] = 0.0
    kronecker[2, 1] = 0.0
    kronecker[2, 2] = 1.0
    ucrossu = np.zeros(shape=(3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            ucrossu[i, j] = u[i] * u[j]
    u_x = np.zeros(shape=(3, 3))
    u_x[0, 0] = 0.0
    u_x[0, 1] = (-1.0) * u[2]
    u_x[0, 2] = u[1]
    u_x[1, 0] = u[2]
    u_x[1, 1] = 0.0
    u_x[1, 2] = (-1.0) * u[0]
    u_x[2, 0] = (-1.0) * u[1]
    u_x[2, 1] = u[0]
    u_x[2, 2] = 0.0
    rotation = np.zeros(shape=(3, 3))
    rotation = ucrossu + np.cos(angle) * (kronecker - ucrossu) + np.sin(angle) * u_x
    return rotation


with open("/mnt/home/jhunt/ceph/Bonsai/r2/B2/step702.p", "rb") as f:
    idd, x, y, z, vx, vy, vz, mass = pickle.load(f)

discindx = mass < 1e-7
bulgeindx = (mass > 1e-7) * (mass < 1e-6)
sgrindx = mass > 1e-6

mx = np.median(x[bulgeindx])
my = np.median(y[bulgeindx])
mz = np.median(z[bulgeindx])
mvx = np.median(vx[bulgeindx])
mvy = np.median(vy[bulgeindx])
mvz = np.median(vz[bulgeindx])

timefw = (702 + 1) * 9.778145 / 1000.0

x = x - mx
y = y - my
z = z - mz
vx = vx - mvx
vy = vy - mvy
vz = vz - mvz

rr = np.sqrt(
    x[discindx] ** 2 + y[discindx] ** 2 + z[discindx] ** 2
)  # distance of stars
Jdisk = np.zeros((3, len((x[discindx]))))
Jdisk[0, :] = (y[discindx]) * (vz[discindx]) - (z[discindx]) * (vy[discindx])
Jdisk[1, :] = (z[discindx]) * (vx[discindx]) - (x[discindx]) * (vz[discindx])
Jdisk[2, :] = (x[discindx]) * (vy[discindx]) - (y[discindx]) * (vx[discindx])
rcyl = 5.0
wr4kpc = np.where((rr < rcyl))
jtot = np.zeros(3)
jtot[0] = np.sum(Jdisk[0, wr4kpc])
jtot[1] = np.sum(Jdisk[1, wr4kpc])
jtot[2] = np.sum(Jdisk[2, wr4kpc])
magjtot = np.sqrt(jtot[0] ** 2 + jtot[1] ** 2 + jtot[2] ** 2)
jtot_normalised = jtot / magjtot
rot_align_J = rotation_axis_angle(jtot_normalised)

post = np.vstack((x, y, z))
velt = np.vstack((vx, vy, vz))

post = np.matmul(rot_align_J, post).T
velt = np.matmul(rot_align_J, velt).T
post[:, 2] *= -1.0
post[:, 0] *= -1.0
velt[:, 2] *= -1.0
velt[:, 0] *= -1.0

fxx = np.argmax(x)
oxx = x[fxx]
oyy = y[fxx]

# oxx=np.median(x[sgrindx])
# oyy=np.median(y[sgrindx])

# xxx=np.median(post[:,0][sgrindx])
# yyy=np.median(post[:,1][sgrindx])

xxx = post[:, 0][fxx]
yyy = post[:, 1][fxx]

dummy, ang1, dummy2 = coords.rect_to_cyl(oxx, oyy, 0.0)
dummy, ang2, dummy2 = coords.rect_to_cyl(xxx, yyy, 0.0)

ang = ang1 - ang2

rotx = post[:, 0] * np.cos(ang) - post[:, 1] * np.sin(ang)
roty = post[:, 0] * np.sin(ang) + post[:, 1] * np.cos(ang)
rotvx = velt[:, 0] * np.cos(ang) - velt[:, 1] * np.sin(ang)
rotvy = velt[:, 0] * np.sin(ang) + velt[:, 1] * np.cos(ang)

rotz = post[:, 2]
rotvz = velt[:, 2]

tbl = at.QTable()
tbl["ID"] = idd[discindx].astype(int)
tbl["xyz"] = np.stack((rotx[discindx], roty[discindx], rotz[discindx])).T.astype("f4")
tbl["vxyz"] = np.stack((rotvx[discindx], rotvy[discindx], rotvz[discindx])).T.astype(
    "f4"
)
tbl.sort("ID")

act_file = "/mnt/home/jhunt/ceph/Bonsai/r2/B2/FlattenedDiscActions/MedianFlattenedDiskActions702.npy"  # noqa
tmp = np.load(act_file)

true_act = at.QTable()
true_act["act_ID"] = tmp[9].astype(int)
true_act["J"] = tmp[:3].T.astype("f4")
true_act["Omega"] = tmp[6:9].T.astype("f4")
true_act["theta"] = tmp[3:6].T.astype("f4")
true_act.sort("act_ID")

act_file = "/mnt/home/jhunt/ceph/Bonsai/r2/B2/FlattenedDiscActions/MedianFlattenedDiskActions100.npy"  # noqa
tmp = np.load(act_file)

init_act = at.QTable()
init_act["init_ID"] = tmp[9].astype(int)
init_act["init_J"] = tmp[:3].T.astype("f4")
init_act["init_Omega"] = tmp[6:9].T.astype("f4")
init_act.sort("init_ID")

tbl = at.hstack((tbl, true_act, init_act))
tbl["R"] = np.sqrt(tbl["xyz"][:, 0] ** 2 + tbl["xyz"][:, 1] ** 2)
tbl["phi"] = np.arctan2(tbl["xyz"][:, 1], tbl["xyz"][:, 0])
tbl["v_R"] = (
    tbl["xyz"][:, 0] * tbl["vxyz"][:, 0] + tbl["xyz"][:, 1] * tbl["vxyz"][:, 1]
) / tbl["R"]
tbl["v_phi"] = (
    tbl["xyz"][:, 0] * tbl["vxyz"][:, 1] - tbl["xyz"][:, 1] * tbl["vxyz"][:, 0]
) / tbl["R"]

tbl.write("../../data/Jason-r2-B2-disk.fits", overwrite=True)
