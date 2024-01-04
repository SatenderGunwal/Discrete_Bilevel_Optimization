from pyscipopt import Model, SCIP_PARAMSETTING, quicksum

import numpy as np
import pickle as pkl
from constraint_handler import Constraint_Handler
# import os

def GBP_CUT( problem_data ):

    [Cx, Cy, Gx, Gy, G0, A, B, C, Dy, problem_senses ] = problem_data
    XDim, YDim                                         = len(Cx), len(Dy)

    model = Model("General Binary Bilevel Solver")
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    model.setSeparating(SCIP_PARAMSETTING.OFF)
    model.setParams({"constraints/linear/upgrade/logicor":False, "constraints/linear/upgrade/indicator":False, 
                    "constraints/linear/upgrade/knapsack":False, "constraints/linear/upgrade/setppc":False, "constraints/linear/upgrade/xor":False,
                    "constraints/linear/upgrade/varbound":False})

    param_dict1 = {"propagating/dualfix/maxprerounds":0,"propagating/genvbounds/maxprerounds":0,"propagating/obbt/maxprerounds":0,
                  "propagating/nlobbt/maxprerounds":0, "propagating/probing/maxprerounds":0,"propagating/pseudoobj/maxprerounds":0,"propagating/redcost/maxprerounds":0,
                  "propagating/rootredcost/maxprerounds":0,"propagating/symmetry/maxprerounds":0,"propagating/vbounds/maxprerounds":0,"constraints/cardinality/maxprerounds":0,
                  "constraints/SOS1/maxprerounds":0,"constraints/SOS2/maxprerounds":0,"constraints/varbound/maxprerounds":0,"constraints/knapsack/maxprerounds":0,
                  "constraints/setppc/maxprerounds":0,"constraints/linking/maxprerounds":0, "constraints/or/maxprerounds":0,"constraints/and/maxprerounds":0,
                  "constraints/xor/maxprerounds":0,"constraints/conjunction/maxprerounds":0,"constraints/disjunction/maxprerounds":0,"constraints/linear/maxprerounds":0,
                  "constraints/orbisack/maxprerounds":0,"constraints/orbitope/maxprerounds":0,"constraints/symresack/maxprerounds":0,"constraints/logicor/maxprerounds":0,
                  "constraints/bounddisjunction/maxprerounds":0,"constraints/cumulative/maxprerounds":0,"constraints/nonlinear/maxprerounds":0,"constraints/pseudoboolean/maxprerounds":0,
                  "constraints/superindicator/maxprerounds":0,"constraints/indicator/maxprerounds":0,"constraints/components/maxprerounds":0,
                 "propagating/maxrounds":0,"propagating/maxroundsroot":0}
    model.setParams(param_dict1)

    model.hideOutput(False)

    param_dict2 = { "presolving/maxrounds":0, "presolving/maxrestarts":0, "presolving/trivial/maxrounds":0, "presolving/inttobinary/maxrounds":0, "presolving/gateextraction/maxrounds":0,
                  "presolving/dualcomp/maxrounds":0, "presolving/domcol/maxrounds":0, "presolving/implics/maxrounds":0, "presolving/sparsify/maxrounds":0, "presolving/dualsparsify/maxrounds":0,
                  }
    model.setParams(param_dict2)
    model.setParam('limits/time', 2)

    X = { f"{i}":model.addVar(vtype="B", name=f"X{i}") for i in range(XDim) }
    Y = { f"{i}":model.addVar(vtype="B", name=f"Y{i}") for i in range(YDim) }

    # Objective
    model.setObjective( quicksum( Cx[idx]*X[f'{idx}'] for idx in range(XDim) ) + quicksum( Cy[idx]*Y[f'{idx}'] for idx in range(YDim) ), 'minimize' )

    # Original Leader's and Follower's Constraints
    num_leader_cons   = len(Gx)
    num_follower_cons = len(A)
    for row in range(num_leader_cons):
        model.addCons( quicksum( Gx[row][idx]*X[f'{idx}'] for idx in range(XDim)) + quicksum( Gy[row][idx]*Y[f'{idx}'] for idx in range(YDim)) <= G0[row], name='ULC' )
    for row in range(num_follower_cons):
        model.addCons( quicksum( A[row][idx]*X[f'{idx}'] for idx in range(XDim)) + quicksum( B[row][idx]*Y[f'{idx}'] for idx in range(YDim)) <= C[row], name='LLC' )

    conshdlr      = Constraint_Handler(X,Y)
    conshdlr.data = problem_data, XDim, YDim, []
    model.includeConshdlr(conshdlr,  "GBP", "Constraint handler for GBP", needscons = False) # needscons: "should the constraint handler be skipped, if no constraints are available?"

    model.optimize()
    obj = model.getObjVal()

    return obj

if __name__=="__main__":

    results = []
    Num_Tests = 1
    for test in range(Num_Tests):
            for id_ in range(2,3):

                file_path = f".\\GBP_10_10_5_5\\GBP_10_10_5_5_{id_}.pkl"

                with open(file_path, 'rb') as file:
                    data = pkl.load(file)

                # Changing problem to Min-Min
                [Cx, Cy, Gx, Gy, G0, A, B, C, Dy, senses] = data
                    
                # HPR Constraint Matrix Rank Check
                B1 = np.hstack((Gx,Gy))
                B2 = np.hstack((A,B))
                M  = np.vstack((B1,B2))
                rank = np.linalg.matrix_rank(M)
                # print("Rank of HPR Matrix = ", rank)

                # Changing Problem to Min-Min form
                if senses[0] == -1:
                    Cx, Cy = -Cx, -Cy
                    senses[0] = 1
                if senses[1] == -1:
                    Dy = -Dy
                    senses[1] = 1
                data = [Cx, Cy, Gx, Gy, G0, A, B, C, Dy, senses]
                #----------------------------------------------

                obj = GBP_CUT(data)
                
                results.append(obj)
    
                print("\n\n",results)