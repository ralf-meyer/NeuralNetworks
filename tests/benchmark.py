import time
from NeuralNetworks.SymmetryFunctionSet_new import SymmetryFunctionSet_py as SymFunSet_py, SymmetryFunctionSet as SymFunSet_cpp
from NeuralNetworks.SymmetryFunctionSetC import SymmetryFunctionSet as SymFunSet_c
import numpy as np

geo55 = [["Ni", np.array([-0.0000014484, 1.4664110123, 2.3723640186])],
 ["Au", np.array([-0.0000161717, 0.0260399013, 0.0422067291])],
 ["Au", np.array([1.4379242875, 2.3608655490, 0.0515513117])],
 ["Au", np.array([2.3283067104, 0.0328066359, 1.4905028650])],
 ["Au", np.array([-1.4395836514, 2.3598646817, 0.0522201938])],
 ["Au", np.array([-2.3276795947, 0.0312360392, 1.4914963357])],
 ["Au", np.array([0.0009883587, -1.4066359652, 2.3805068195])],
 ["Au", np.array([-1.4493787582, -2.3486947327, 0.0022040854])],
 ["Au", np.array([1.4530803114, -2.3463949757, 0.0007948156])],
 ["Au", np.array([2.3475096093, 0.0027997949, -1.4510857794])],
 ["Au", np.array([-0.0024016161, 1.4522009968, -2.3471493029])],
 ["Au", np.array([-2.3489551793, -0.0009308170, -1.4488298166])],
 ["Au", np.array([0.0000710545, -1.4548338762, -2.3542016393])],
 ["Au", np.array([0.0001150205, 2.8469621106, 4.6066635718])],
 ["Au", np.array([1.4577260452, 3.8487451318, 2.4157838738])],
 ["Au", np.array([2.3562489072, 1.4912534949, 3.8728502088])],
 ["Au", np.array([2.8820390027, 4.6684400067, 0.0049456846])],
 ["Au", np.array([3.9517087033, 2.4162752762, 1.4646131139])],
 ["Au", np.array([4.6648620132, 0.0036814147, 2.8876216541])],
 ["Au", np.array([-1.4551943797, 3.8502354864, 2.4149011622])],
 ["Au", np.array([-0.0013529959, 4.8584943427, -0.0447398180])],
 ["Au", np.array([-2.8833847172, 4.6676743304, 0.0054450284])],
 ["Au", np.array([-2.3570633876, 1.4938296316, 3.8712880915])],
 ["Au", np.array([-3.9524162559, 2.4142965316, 1.4661197694])],
 ["Au", np.array([-4.6644223259, 0.0025271386, 2.8884278735])],
 ["Au", np.array([-0.0015183854, 0.0361737682, 4.7727659745])],
 ["Au", np.array([-2.4416650544, -1.5375981115, 3.9088291364])],
 ["Au", np.array([0.0007305770, -2.8795196626, 4.6703074807])],
 ["Au", np.array([2.4438090740, -1.5366280445, 3.9079429103])],
 ["Au", np.array([-1.4829104477, -3.9093189520, 2.4182661722])],
 ["Au", np.array([1.4893041186, -3.9057360748, 2.4171931342])],
 ["Au", np.array([-2.8750170741, -4.6658754504, -0.0056687253])],
 ["Au", np.array([0.0004544458, -4.8422183942, 0.0210197923])],
 ["Au", np.array([2.8830449422, -4.6608657860, -0.0087641742])],
 ["Au", np.array([3.9131562936, -2.4074618484, 1.4900311476])],
 ["Au", np.array([4.8284793533, -0.0008207786, 0.0037942457])],
 ["Au", np.array([3.9276411034, -2.4141258339, -1.4794102714])],
 ["Au", np.array([4.6571613804, -0.0003326597, -2.8887231694])],
 ["Au", np.array([3.9013649307, 2.4230441129, -1.4955766700])],
 ["Au", np.array([1.4942367159, 3.9073580158, -2.4116143559])],
 ["Au", np.array([2.4270293568, 1.5134841481, -3.9070420188])],
 ["Au", np.array([-0.0051163197, 2.8747160188, -4.6662357294])],
 ["Au", np.array([-1.5021376693, 3.9059385232, -2.4120143772])],
 ["Au", np.array([-3.9047751167, 2.4176182587, -1.4908773466])],
 ["Au", np.array([-2.4279580727, 1.5130785901, -3.9067307257])],
 ["Au", np.array([-4.6602382707, -0.0085586969, -2.8838414301])],
 ["Au", np.array([-4.8298762435, -0.0076364199, 0.0068198647])],
 ["Au", np.array([-3.9075847028, -2.4112240377, 1.4936771507])],
 ["Au", np.array([-3.9272849564, -2.4148703936, -1.4788908725])],
 ["Au", np.array([0.0000939704, -2.8864777443, -4.6711854979])],
 ["Au", np.array([0.0338385585, -0.0043974673, -4.8454950483])],
 ["Au", np.array([2.4294287174, -1.5262848317, -3.9045705673])],
 ["Au", np.array([1.4677913680, -3.9351054590, -2.4158433297])],
 ["Au", np.array([-1.5218424129, -3.9016237632, -2.4365474244])],
 ["Au", np.array([-2.4083831653, -1.4719099067, -3.9381161552])]]



geo = [["Ni", np.array([0.0, 0.0, 0.0])],
        ["Ni", np.array([1.2, 0.0, 0.0])],
        ["Au", np.array([0.0, 1.2, 0.0])],
        ["Ni", np.array([-1.2, 0.0, 0.0])],
        ["Au", np.array([0.0, -1.2, 0.0])],
        ["Au", np.array([0.0, 0.0, 1.2])],
        ["Ni", np.array([0.0, 0.0, -1.2])],
        ["Ni", np.array([1.2, 1.2, 0.0])],
        ["Ni", np.array([-1.2, 1.2, 0.0])],
        ["Ni", np.array([1.2, -1.2, 0.0])],
        ["Ni", np.array([-1.2, -1.2, 0.0])],
        ["Ni", np.array([1.2, 1.2, 1.2])],
        ["Ni", np.array([-1.2, 1.2, 1.2])],
        ["Au", np.array([1.2, -1.2, 1.2])],
        ["Ni", np.array([-1.2, -1.2, 1.2])],
        ["Ni", np.array([1.2, 1.2, -1.2])],
        ["Ni", np.array([-1.2, 1.2, -1.2])],
        ["Au", np.array([1.2, -1.2, -1.2])],
        ["Ni", np.array([-1.2, -1.2, -1.2])],
        ["Ni", np.array([1.2, 1.2, 2.4])],
        ["Ni", np.array([-1.2, 1.2, 2.4])],
        ["Au", np.array([1.2, -1.2, 2.4])],
        ["Ni", np.array([-1.2, -1.2, 2.4])],
        ["Ni", np.array([1.2, 1.2, -2.4])],
        ["Ni", np.array([-1.2, 1.2, -2.4])],
        ["Au", np.array([1.2, -1.2, -2.4])],
        ["Ni", np.array([-1.2, -1.2, -2.4])],
        ["Ni", np.array([1.2, 1.2, 3.6])],
        ["Ni", np.array([-1.2, 1.2, 3.6])],
        ["Au", np.array([1.2, -1.2, 3.6])],
        ["Ni", np.array([-1.2, -1.2, 3.6])]]

rss = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
etas = [1.0]

zetas = [1.0, 2.0, 3.0]
lambdas = [1.0, 2.0, 3.0, 4.0]

sfs_cpp = SymFunSet_cpp(["Ni", "Au"])
sfs_cpp.add_radial_functions(rss, etas)
sfs_cpp.add_angular_functions([1.0], zetas, lambdas)

sfs_py = SymFunSet_py(["Ni", "Au"])
sfs_py.add_radial_functions(rss, etas)
sfs_py.add_angular_functions([1.0,], zetas, lambdas)

sfs_c = SymFunSet_c(["Ni", "Au"])
sfs_c.add_radial_functions(rss, etas)
sfs_c.add_angular_functions([1.0], zetas, lambdas)

start_time = time.time()
out_cpp = sfs_cpp.eval_geometry(geo55)
end_time = time.time()
print("Cpp:      {} s".format(end_time-start_time))

start_time = time.time()
out_cpp_new = sfs_cpp.eval_geometry_old(geo55)
end_time = time.time()
print("Cpp_old:  {} s".format(end_time-start_time))

start_time = time.time()
out_py = sfs_py.eval_geometry(geo55)
end_time = time.time()
print("Python:   {} s".format(end_time-start_time))

start_time = time.time()
out_c = np.asarray(sfs_c.eval_geometry(geo55))
end_time = time.time()
print("C:        {} s".format(end_time-start_time))

print("Difference of Python and Cpp < 1e-6: {}".format(all(abs(out_cpp-out_py.flatten()) < 1e-6)))

print("\n--- Derivatives ---")

types = [a[0] for a in geo55]
xyzs = np.array([a[1] for a in geo55])

start_time = time.time()
out_cpp = sfs_cpp.eval_derivatives(types, xyzs)
end_time = time.time()
print("Cpp:      {} s".format(end_time-start_time))

start_time = time.time()
out_cpp_new = sfs_cpp.eval_derivatives_old(types, xyzs)
end_time = time.time()
print("Cpp_old:  {} s".format(end_time-start_time))
