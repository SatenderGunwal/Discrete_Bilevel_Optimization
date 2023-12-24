import numpy as np

class Intersection_Cuts:

    def __init__(self, A, B, C, problem_data, HPR_Solution_X, HPR_Solution_Y, Optimal_Y, BasisIndexes ):
        self.Cx = problem_data[0]
        self.Cy = problem_data[1]
        self.Gx = problem_data[2]
        self.Gy = problem_data[3]
        self.G0 = problem_data[4]
        self.A  = problem_data[5]
        self.B  = problem_data[6]
        self.C  = problem_data[7]
        self.ANew  = A
        self.BNew  = B
        self.CNew  = C
        self.Dy = problem_data[8]
        self.Optimal_Y      = Optimal_Y
        self.problem_senses = problem_data[9]
        self.HPR_Solution_X = HPR_Solution_X
        self.HPR_Solution_Y = HPR_Solution_Y
        self.BasisIndexes   = BasisIndexes

    def Bilevel_Free_Disjunction(self): # OptY

        OptY  = self.Optimal_Y
        if OptY == str: return "FMILP Error"

        DyHat            = self.Dy @ np.array(OptY)
        Bilevel_Free_RHS = self.C + np.ones((len(self.C))) - self.B @ np.array(OptY)

        # Here, it is checked if (x*,y*) [Incumbent HPR Soloution] belongs to interior of bievel free set S or not.
        #......................................
        # X-Related Constraints Violation Check
        row_num = 0
        violated_row_indexes = []
        for row in self.A:
            if row @ self.HPR_Solution_X <= Bilevel_Free_RHS[row_num]:
                violated_row_indexes.append(row_num)
            row_num += 1
        A_subarray = np.take(self.A, violated_row_indexes, axis=0) # Getting Violated A rows

        # Y-Related Constraint Violation Check
        XDim, YDim        = len(self.Cx), len(self.Cy)
        Y_constr_violated = False
        if self.Dy @ self.HPR_Solution_Y >= DyHat:
            Y_constr_violated = True

        # _______ Compiling Violated Row Data from bilevel free set ______________
        num_violated_x_constr = A_subarray.shape[0]
        A_subarray = np.hstack( (A_subarray, np.zeros((num_violated_x_constr, YDim))) ) # Adding zero aray block
        if Y_constr_violated == True:
            y_row     = np.hstack( (np.zeros(XDim), -1*self.Dy) )
            Final_LHS = np.concatenate((np.array([list(y_row)]), A_subarray), axis=0)
            Final_RHS = np.array([-1*DyHat] + list(Bilevel_Free_RHS))
        else:
            Final_LHS = A_subarray
            Final_RHS = Bilevel_Free_RHS

        return Final_LHS, Final_RHS

    def ABHatMatrix(self):

        Block_1 = np.concatenate((self.Gx, self.Gy), axis=1)
        Block_2 = np.concatenate((self.ANew, self.BNew), axis=1)
        A_hat   = np.concatenate((Block_1, Block_2), axis=0)
        B_hat   = np.take(A_hat, self.BasisIndexes, axis=1)

        return A_hat, B_hat

    def Cut(self):

        XDim, YDim           = len(self.Cx), len(self.Cy)
        Final_LHS, Final_RHS = self.Bilevel_Free_Disjunction()
        A_hat, B_hat         = self.ABHatMatrix()
        Combined_RHS         = np.concatenate((self.G0, self.CNew), axis=0)
        B_hat_pseudoinverse  = np.linalg.inv(B_hat)

        # Code Line 1-3 from Algorithm 1(IC Separation) from Fischetti
        num_disjunctions = Final_LHS.shape[0]
        for row_num in range(num_disjunctions):
            older_lhs = np.array([list(Final_LHS[row_num])])#Final_LHS[row_num]
            older_rhs = Final_RHS[row_num]
            U_i       = np.take(older_lhs, self.BasisIndexes, axis=1) @ B_hat_pseudoinverse
            Final_LHS[row_num] = older_lhs - U_i @ A_hat
            Final_RHS[row_num] = older_rhs - U_i @ Combined_RHS

        # Code Line 4
        gamma = []
        for idx in range( XDim + YDim ):
            column_   = np.take(Final_LHS, idx,axis=1)
            coef_list = [ column_[i]/Final_RHS[i] for i in range(num_disjunctions) ]
            gamma.append(max(coef_list))

        # Code line 5-9 (Skipping Line 7 by assuming all variables are integers)
        if all(x >= 0 for x in gamma):
            for i in range(XDim + YDim):
                gamma[i] = min([gamma[i],1])
        # Code line 10
        return gamma