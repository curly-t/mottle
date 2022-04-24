import ndtamr.NDTree as nd
import ndtamr.AMR as amr
import ndtamr.Vis as vis
from ndtamr.Data import GenericData
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from glob import glob
import pickle
from ast import literal_eval        # Parse tuples
from os import path


def define_func_class(field_func):
    # The input field_func(rho, z)
    class FuncClass(GenericData):
        """
        Class object for a function, which wraps the inputed function for the AMR code.
        """
        data_cols = ['value']

        def __init__(self, coords=(0, 0), file=None, data=None):
            GenericData.__init__(self, coords=coords, file=file, data=data)

        def func(self):
            """Function which sets the data value"""
            res = field_func(*self.coords)
            if np.isnan(res) or np.isinf(res):
                res = 1
            return res

        def get_refinement_data(self):
            """Returns the data column which we want to refine on."""
            return self.value

    return FuncClass


def calc_far_bound_points(field_func, bbox, num_bound_points):
    """ This function is needed because the tree includes the low bound, but excludes the high bound,
    making the points near the high bound possibly undefined! """

    print(bbox, num_bound_points)
    # Ena stranica končnih točk
    rhos1 = np.ones(num_bound_points//2 - 1) * bbox[1][0]
    zs1 = np.linspace(bbox[0][1], bbox[1][1], num=num_bound_points//2 - 1, endpoint=False)

    # Druga stranica končnih točk
    zs2 = np.ones(num_bound_points//2) * bbox[1][1]
    rhos2 = np.linspace(bbox[0][0], bbox[1][0], num=num_bound_points//2, endpoint=True)

    rhos = np.concatenate((rhos1, rhos2))
    zs = np.concatenate((zs1, zs2))

    vals = np.zeros(2 * (num_bound_points//2) - 1)
    for i in range(2 * (num_bound_points//2) - 1):
        vals[i] = field_func(rhos[i], zs[i])

    return np.array([rhos, zs]).T, vals


def make_initial_tree(field_func, bbox, initial_depth):
    # bbox = ((rho_min, z_min), (rho_max, z_max))
    FuncClass = define_func_class(field_func)
    t = nd.make_uniform(depth=initial_depth,
                        dim=2,
                        data_class=FuncClass,
                        xmin=bbox[0], xmax=bbox[1],
                        restrict_func=nd.restrict_datafunc, prolongate_func=nd.prolongate_datafunc)
    return t


def get_coords_and_vals_from_tree(tree):
    leaves = tree.list_leaves(attr="data")
    coords = np.array([leaf.coords for leaf in leaves])     # shape = (N, 2)
    values = np.array([leaf.value for leaf in leaves])      # shape = (N)
    return coords, values


def construct_interpolation(coords, values, fill_value=0.):
    return CloughTocher2DInterpolator(coords, values, fill_value=fill_value, maxiter=10000, tol=1e-10)


def test_interpolation(interp_func, field_func, bbox, num_test_points):
    rhos = np.random.random(num_test_points) * (bbox[1][0] - bbox[0][0]) + bbox[0][0]
    zs = np.random.random(num_test_points) * (bbox[1][1] - bbox[0][1]) + bbox[0][1]

    print("Calculating field_points!")
    # Usually field_func just takes single floats as inputs - so it needs to be fed floats
    field_vals = np.zeros(num_test_points)
    for i in range(num_test_points):
        field_vals[i] = field_func(rhos[i], zs[i])

    # But interp_func can take vectorized input
    interp_vals = interp_func(rhos, zs)

    devs = interp_vals - field_vals
    max_dev = np.max(np.abs(devs))
    median_dev = np.median(np.abs(devs))
    std_dev = np.std(np.abs(devs))
    perc95 = np.percentile(np.abs(devs), 95)
    perc99 = np.percentile(np.abs(devs), 99)
    perc999 = np.percentile(np.abs(devs), 99.9)
    print(f"Maximum delta: {max_dev}")
    print(f"Median delta: {median_dev}")
    print(f"Sdev: {std_dev}")
    print(f"Percentiles: 95: {perc95}\t99: {perc99}\t99.9: {perc999}")

    return (max_dev, median_dev, std_dev), ((rhos, zs), devs)


def refine_tree(tree, field_func, bbox, far_bound_points, ref_tol=0.1, ref_extent=4, ref_min_val=1e-5, ref_eps=0.,
                final_ref_tol=1e-3, max_ref_depth=12, num_test_points=1000):
    num_steps = max_ref_depth - tree.depth()
    for i in range(num_steps):
        amr.refine(tree, tol=ref_tol, eps=ref_eps, extent=ref_extent, min_value=ref_min_val)    # Refine
        amr.compression(tree)

        coords, vals = get_coords_and_vals_from_tree(tree)
        full_coords, full_vals = np.concatenate((coords, far_bound_points[0])), np.concatenate((vals, far_bound_points[1]))
        interp_func = construct_interpolation(full_coords, full_vals)
        stat_vals, deviations = test_interpolation(interp_func, field_func, bbox, num_test_points)

        vis.plot(tree, grid=True, cmap="Greys", aspect="auto", colorbar=False)
        plt.scatter(deviations[0][0], deviations[0][1], alpha=0.05 + 0.95*np.abs(deviations[1])/np.max(np.abs(deviations[1])), marker=".")
        plt.show()

        if stat_vals[0] < final_ref_tol:
            # Asks to break when the refinement tolerance is reached
            ans = str(input("Refinement tolerance has been reached. Stop refinement? [Y/n] "))
            if ans != "n":
                print("Stopping refinement!")
                break
    print("Returning interp func!")
    return interp_func


def interpolate_func(func, bbox=None, initial_depth=None, num_of_far_bound_points=2000, **kwargs):
    far_bound_points = calc_far_bound_points(func, bbox, num_of_far_bound_points)
    tree = make_initial_tree(func, bbox, initial_depth)
    return refine_tree(tree, func, bbox, far_bound_points, **kwargs)


def construct_save_name(func_num, interp_params, external_params):
    name_str = func_num
    interp_param_str = "_".join(
        [f"{str(key).replace('_', ':')}={str(type(val))[8]}{str(val).replace('.', ':')}" for (key, val) in
         interp_params.items()])
    extrn_param_str = "_".join(
        [f"{str(key).replace('_', ':')}={str(type(val))[8]}{str(val).replace('.', ':')}" for (key, val) in
         external_params.items()])
    return "__".join([name_str, interp_param_str, extrn_param_str]) + ".interp_func"


def parse_save_name(filename):
    stripped = filename[:-12]
    name, interp_param_str, extern_param_str = tuple(stripped.split("__"))
    interp_param_strings = interp_param_str.split("_")
    extern_param_strings = extern_param_str.split("_")

    def parse_param_strings(param_strings):
        if param_strings == [""]:
            return {}

        params_dict = {}
        for param_str in param_strings:
            key, val = tuple(param_str.split("="))
            key = key.replace(":", "_")

            val_type = val[0]
            if val_type == "i":
                val = int(val[1:])
            elif val_type == "f":
                val = float(val[1:].replace(":", "."))
            elif val_type == "t":
                # Tuple
                val = literal_eval(val[1:].replace(":", "."))

            params_dict.update({key: val})
        return params_dict

    return {"filename": filename, "name": name,
            "interp_params": parse_param_strings(interp_param_strings),
            "external_params": parse_param_strings(extern_param_strings)}


def parse_all_saved_interps_names():
    filenames = [path.split(fname)[1] for fname in list(glob(path.join("interpolated_fields", "*.interp_func")))]
    properties = []
    for filename in filenames:
        properties.append(parse_save_name(filename))
    return properties


def search_saved_interps_for_match(func, interp_params, external_params, all_saved_interpolators):
    for saved_interpolator in all_saved_interpolators:
        if func.__name__ == saved_interpolator["name"]:
            if interp_params == saved_interpolator["interp_params"] and external_params == saved_interpolator["external_params"]:
                return saved_interpolator["filename"]


def load_saved_interpolator(filename):
    with open(path.join("interpolated_fields", filename), "rb") as f:
        interpolator = pickle.load(f)
    return interpolator


def save_interpolator(interpolator, func_name, interp_params, external_params):
    filename = construct_save_name(func_name, interp_params, external_params)
    with open(path.join("interpolated_fields", filename), "wb") as f:
        print("Dumping!")
        pickle.dump(interpolator, f)


def get_interpolated_func(func, interp_params, external_params={}):
    all_saved_interpolators = parse_all_saved_interps_names()
    interp_filename = search_saved_interps_for_match(func, interp_params, external_params, all_saved_interpolators)
    if interp_filename is not None:
        return load_saved_interpolator(interp_filename)
    else:
        # New one will have to be calculated
        interp_func = interpolate_func(func, **interp_params)
        save_interpolator(interp_func, func.__name__, interp_params, external_params)
        return interp_func


if __name__ == "__main__":
    def external_func(rho, z):
        return rho ** 2 - rho + np.sin(z * rho * 2)

    interp_params = {"bbox": ((-1, -1), (3, 3)), "initial_depth": 4, "num_of_far_bound_points": 2000}
    get_interpolated_func(external_func, interp_params)

