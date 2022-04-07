import ndtamr.NDTree as nd
import ndtamr.AMR as amr
import ndtamr.Vis as vis
from ndtamr.Data import GenericData
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator


def external_func(rho, z):
    return rho**2 - rho + np.sin(z*rho*2)


class MyFuncClass(GenericData):
    """
    2D test class which consists of a one-armed spiral.
    """
    data_cols = ['value']

    def __init__(self, coords=(0, 0), file=None, data=None):
        GenericData.__init__(self, coords=coords, file=file, data=data)

    def func(self):
        """Function which sets the data value"""
        res = external_func(*self.coords)
        if np.isnan(res) or np.isinf(res):
            res = 1
        return res

    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value

def func(xc, yc):
    r = np.sqrt( xc**2 + yc**2)
    p = np.arctan2(yc,xc)

    ps = np.log(r/1)/.2
    xs = r*np.cos(ps)
    ys = r*np.sin(ps)
    res = np.exp(-((xc-xs)**2 + (yc-ys)**2)/(2*.3**2))
    return res

class SpiralTest2D(GenericData):
    """
    2D test class which consists of a one-armed spiral.
    """
    data_cols = ['value']
    def __init__(self,coords=(0,0),file=None,data=None):
        GenericData.__init__(self,coords=coords,file=file,data=data)

    def func(self):
        """Function which sets the data value"""
        xc,yc = self.coords
        r = np.sqrt( xc**2 + yc**2)
        p = np.arctan2(yc,xc)

        ps = np.log(r/1)/.2
        xs = r*np.cos(ps)
        ys = r*np.sin(ps)
        res = np.exp(-((xc-xs)**2 + (yc-ys)**2)/(2*.3**2))
        if np.isnan(res) or np.isinf(res):
            res = 1
        return res
    def get_refinement_data(self):
        """Returns the data column which we want to refine on."""
        return self.value



t = nd.make_uniform(depth=4,
                    dim=2,
                    data_class=SpiralTest2D,
                    xmin=(-2, -2), xmax=(2, 2),
                    restrict_func=nd.restrict_datafunc, prolongate_func=nd.prolongate_datafunc)

vis.plot(t, grid=True)
plt.show()

for i in range(8):
    # amr.refine(t, tol=0.2, eps=0., extent=4, min_value=1e-4)   # Spiral thingy test
    # amr.refine(t, tol=0.05, eps=0., extent=4, min_value=1e-4)   # Worked well for spiral thingy
    amr.refine(t, tol=0.6, eps=0., extent=5, min_value=1e-5)   # MyFunc
    amr.compression(t)
    vis.plot(t, grid=True, cmap="Greys")
    plt.show()

    leaves = t.list_leaves(attr="data")
    coords = np.array([leaf.coords for leaf in leaves])
    values = np.array([leaf.value for leaf in leaves])

    # Fitting
    interp = CloughTocher2DInterpolator(coords, values)
    xs = np.linspace(-2, 2, num=1000)
    ys = np.linspace(-2, 2, num=1000)
    Xs, Ys = np.meshgrid(xs, ys)
    Xs_flat, Ys_flat = np.concatenate(Xs), np.concatenate(Ys)
    coords_i = np.array([Xs_flat, Ys_flat]).T
    spl_rep = interp(coords_i).reshape((1000, 1000))
    print(spl_rep.shape)
    plt.imshow(spl_rep)
    plt.title("Interpolated")
    plt.show()
    # plt.imshow(external_func(Xs, Ys))
    plt.imshow(func(Xs, Ys))
    plt.title("Actual")
    plt.show()
    # col = plt.imshow(spline(xs, ys) - external_func(Xs, Ys))
    col = plt.imshow(spl_rep - func(Xs, Ys))
    plt.colorbar(col)
    plt.title("Discrepancy")
    plt.show()


# leaves = t.list_leaves(attr="data")
# print([(leaf.coords, leaf.value) for leaf in leaves])


