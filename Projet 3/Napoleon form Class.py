import numpy as np
from scipy.stats import norm
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class MonteCarlo:
    """Class that simulates a Monte Carlo simulation for a given asset.

    Attributes:
        S_0 (float): Initial asset price.
        sigma (float): Volatility of the asset.
        mu (float): Drift of the asset.
        Ns (int): Number of simulations.
        ic (float): Confidence interval.
        maturity (int): Maturity of the asset.
        C (float): Coupon of the asset.
        floor (float): Floor price of the asset.

    Methods:
        generate_t(): Generates the time vector.
        simulate_gbm(t): Simulates a GBM trajectory.
        payoff(S): Calculates the payoff of a given trajectory.
        simulate_monte_carlo(): Simulates Ns trajectories and calculates the payoff of each one.
        convergence_mc(P): Calculates the Monte Carlo convergence curve.
        generate_convergence_curve(): Generates the Monte Carlo convergence curve.
    """
    def __init__(self, S_0, sigma, mu, Ns,ic, maturity, C, floor):
        """Initialize MonteCarlo object

        Args:
            S_0 (float): Initial asset price.
            sigma (float): Volatility of the asset.
            mu (float): Drift of the asset.
            Ns (int): Number of simulations.
            ic (float): Confidence interval.
            maturity (int): Maturity of the asset.
            C (float): Coupon of the asset.
            floor (float): Floor price of the asset.
        """
        self.S_0 = S_0
        
        self.sigma = sigma
        self.mu = mu
        self.Ns = Ns
        self.maturity = maturity
        self.C = C
        self.ic = ic
        self.floor = floor

    def generate_t(self):
        """Generates the time vector.

        Returns:
            List : Time vector.
        """
        t = [i for i in range(12*self.maturity)] #12*n car on a 12 mois dans l'année et l'on regarde tous les mois pour le payoff
        return t

    def simulate_gbm(self, t):
        """Simulates a GBM trajectory.

        Args:
            t (List): Time vector.

        Returns:
            List : A GBM trajectory.
        """
        S = [self.S_0]
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            S.append( S[-1] * np.exp( (self.mu-(self.sigma**2)/2)*dt + self.sigma*np.sqrt(dt)*np.random.normal(0, 1) ) )
        return S

    def payoff(self, S):
        """Calculates the payoff of a given trajectory.

        Args:
            S (List): Trajectory.

        Returns:
            _type_: Payoff of the asset.
        """
        payoff = 0
        for i in range(int(len(S)/12)):
            rend_mensuel = []
            for j in range(12):
                rend_mensuel.append(S[i*12+j]/S[i*12+j-1]-1) #Minus 1 because we want the monthly return
            pire_rend_mensuel = min(rend_mensuel)
            payoff += max(self.C+pire_rend_mensuel, self.floor)
        return payoff

    def simulate_monte_carlo(self):
        """Simulates Ns trajectories and calculates the payoff of each one.

        Returns:
            List: The payoff of each trajectory.
        """
        t = self.generate_t()
        P = []
        for i in range(self.Ns):
            S = self.simulate_gbm(t) 
            P.append(self.payoff(S))
        return P
    
    def convergence_mc(self, P):
        """Calculates the Monte Carlo convergence curve.

        Args:
            P (List): The payoff of each trajectory.

        Returns:
            M: Vector of the moving average.
            b_inf: Vector of the lower bound.
            b_sup: Vector of the upper bound.
        """
        a = norm.ppf(self.ic)
        M = []
        ET = []
        b_inf = []
        b_sup = []
        for i in range(len(P)):
            M.append(np.mean(P[:i+1]))
            ET.append(np.std(P[:i+1]))
            b_inf.append( M[-1] - a*ET[-1]/np.sqrt(i) )
            b_sup.append( M[-1] + a*ET[-1]/np.sqrt(i) )
        return M, b_inf, b_sup

    def generate_convergence_curve(self):
        """Generates the Monte Carlo convergence curve.

        Returns:
            Figure: The Monte Carlo convergence curve.
        """
        P = self.simulate_monte_carlo()
        M, b_inf, b_sup = self.convergence_mc(P)
        fig = Figure(figsize=(10, 4))
        plot = fig.add_subplot(111)
        plot.plot(range(self.Ns-1), M[1:], 'g', label = "Moyenne")
        plot.plot(range(self.Ns-1), b_inf[1:], 'r', label = "Borne inférieure")
        plot.plot(range(self.Ns-1), b_sup[1:], 'r', label = "Borne supérieure")
        plot.set_xlabel("Nombre de simulations")
        plot.set_ylabel("Prix")
        final_price = M[-1]
        error = b_sup[-1] - b_inf[-1]
        plot.set_title(f"Courbe d'évolution du prix en fonction du nombre de simulation de Monte Carlo\nFinal price: {final_price:.4f}, Error: {error:.4f}")
        plot.grid()
        plot.legend()
        return fig



class Form:
    """Class that creates a form to display the Monte Carlo convergence curve.

    Attributes:
        form (tkinter.Tk): The form.
        canvas (FigureCanvasTkAgg): The canvas to display the convergence curve.
        toolbar (NavigationToolbar2Tk): The toolbar to interact with the canvas.

    Methods:
        display_convergence_curve(): Displays the Monte Carlo convergence curve.
        reset_form(): Resets the form.
        run(): Runs the form.
    """
    def __init__(self):
        self.form = tk.Tk()
        self.form.title("Monte Carlo Convergence Curve")
        self.form.geometry("800x600")
        self.canvas = None
        self.toolbar = None

        self.S_0_label = tk.Label(self.form, text="S_0:")
        self.S_0_label.pack()
        self.S_0_entry = tk.Entry(self.form)
        self.S_0_entry.pack()

        self.sigma_label = tk.Label(self.form, text="sigma:")
        self.sigma_label.pack()
        self.sigma_entry = tk.Entry(self.form)
        self.sigma_entry.pack()

        self.mu_label = tk.Label(self.form, text="mu:")
        self.mu_label.pack()
        self.mu_entry = tk.Entry(self.form)
        self.mu_entry.pack()

        self.maturity_label = tk.Label(self.form, text="Maturity (in years):")
        self.maturity_label.pack()
        self.maturity_entry = tk.Entry(self.form)
        self.maturity_entry.pack()

        self.Ns_label = tk.Label(self.form, text="Ns:")
        self.Ns_label.pack()
        self.Ns_entry = tk.Entry(self.form)
        self.Ns_entry.pack()

        self.ic_label = tk.Label(self.form, text="ic:")
        self.ic_label.pack()
        self.ic_entry = tk.Entry(self.form)
        self.ic_entry.pack()

        self.c_label = tk.Label(self.form, text="C:")
        self.c_label.pack()
        self.c_entry = tk.Entry(self.form)
        self.c_entry.pack()

        self.floor_label = tk.Label(self.form, text="Floor :")
        self.floor_label.pack()
        self.floor_entry = tk.Entry(self.form)
        self.floor_entry.pack()

        self.generate_button = tk.Button(self.form, text="Generate Convergence Curve", command=self.display_convergence_curve)
        self.generate_button.pack()

        self.reset_button = tk.Button(self.form, text="Reset Form", command=self.reset_form)
        self.reset_button.pack()
        



    def display_convergence_curve(self):
        """Displays the Monte Carlo convergence curve."""
        S_0 = float(self.S_0_entry.get())
 
        sigma = float(self.sigma_entry.get())
        mu = float(self.mu_entry.get())
        maturity = int(self.maturity_entry.get())
        Ns = int(self.Ns_entry.get())
        ic = float(self.ic_entry.get())
        C = float(self.c_entry.get())
        floor = float(self.floor_entry.get())

        mc = MonteCarlo(S_0, sigma, mu, Ns, ic, maturity, C, floor)
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        fig = mc.generate_convergence_curve()
        self.canvas = FigureCanvasTkAgg(fig, master=self.form)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.form)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    def reset_form(self):
        """Resets the form."""
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.pack_forget()
    
    def run(self):
        """Runs the form."""
        self.form.mainloop()


form = Form()
form.run()