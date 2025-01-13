import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,QLabel,
                             QHBoxLayout,QGridLayout,QPushButton,QLineEdit,QSlider, 
                             QComboBox)
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pyqtgraph as pg
from scipy.fft import fft, fftfreq
import pandas as pd
import csv


class SamplingTheorem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sampling Theorem studio")
        self.setGeometry(200,200,1500,1200)
        self.components =[]
        self.generated_signal = None
        self.composer_frequencies=[]
        self.initUI()

    def initUI(self):
        main_window=QWidget()
        container=QGridLayout()
        main_window.setLayout(container)
        self.setCentralWidget(main_window)

        # Initialize four graphs using the create_graph_canvas function
        self.canvas1 = self.create_graph_canvas("Original Signal with Sample Points")
        self.canvas2 = self.create_graph_canvas("Reconstructed")
        self.canvas3 = self.create_error_canvas("Difference")
        self.canvas4 = self.create_graph_canvas("Frequency Domain")
        
        # GRAPHS
        container.addWidget(self.canvas1,0,1)
        container.addWidget(self.canvas2,0,2)
        container.addWidget(self.canvas3,1,1)
        container.addWidget(self.canvas4,1,2)

        # CONTROLS
        controls_layout1=QVBoxLayout()
        controls_layout2=QVBoxLayout()
        self.upload_signal_label=QLabel("Upload signal:")
        self.upload_signal_button=QPushButton("Upload")
        self.upload_signal_button.clicked.connect(self.upload_and_plot_signal)
        self.upload_signal_button.clicked.connect(self.update_freq_label)

        # Reconstruction menu
        reconstruction_method_layout=QHBoxLayout()
        self.reconstruction_label=QLabel("Reconstruction method:")
        self.reconstruction_menu=QComboBox()
        self.reconstruction_menu.addItems(["Whittaker-Shannon", "Zero-Order Hold", "Linear Interpolation"])
        self.reconstruction_menu.currentIndexChanged.connect(self.update_plots)
        reconstruction_method_layout.addWidget(self.reconstruction_label)
        reconstruction_method_layout.addWidget(self.reconstruction_menu)

        # Signal mixer
        self.signal_mixer_label=QLabel("Signal mixer")
        self.freq_signal_label=QLabel("Frequency")
        freq_signal_layout=QHBoxLayout()
        self.freq_slider=QSlider(Qt.Horizontal)
        self.freq_slider.setRange(0, 100)  # Set range for frequency from 1Hz to 100Hz
        self.freq_slider.setValue(0)  # Default value
        self.frequency_value=QLabel("0 HZ")
        self.freq_slider.valueChanged.connect(self.update_frequency_label)
        freq_signal_layout.addWidget(self.freq_slider)
        freq_signal_layout.addWidget(self.frequency_value)
        self.amplitude_signal_label=QLabel("Amplitude")
        amplitude_signal_layout=QHBoxLayout()
        self.amplitude_slider=QSlider(Qt.Horizontal)
        self.amplitude_slider.setRange(0, 100)
        self.amplitude_slider.setValue(0) 
        self.amplitude_value=QLabel("0")
        self.amplitude_slider.valueChanged.connect(self.update_amplitude_label)
        amplitude_signal_layout.addWidget(self.amplitude_slider)
        amplitude_signal_layout.addWidget(self.amplitude_value)
        self.phase_label=QLabel("Phase")
        phase_layout=QHBoxLayout()
        self.phase_slider=QSlider(Qt.Horizontal)
        self.phase_slider.setRange(0,360)
        self.phase_slider.setValue(0)
        self.phase_slider.valueChanged.connect(self.update_phase_label)
        self.phase_value=QLabel("0")
        phase_layout.addWidget(self.phase_slider)
        phase_layout.addWidget(self.phase_value)
        self.add_signal_button=QPushButton("Add")
        self.add_signal_button.clicked.connect(self.add_component)
        self.add_signal_button.clicked.connect(self.update_freq_label)
        self.save_signal_button=QPushButton("Save signal")
        self.save_signal_button.clicked.connect(self.save_signal)
        mixer_buttons_layout=QHBoxLayout()

        mixer_buttons_layout.addWidget(self.add_signal_button)
        mixer_buttons_layout.addWidget(self.save_signal_button)


        # Remove components
        self.remove_signal_label=QLabel("Remove signal components")
        remove_layout=QHBoxLayout()
        self.remove_signals_menu=QComboBox()
        self.remove_button=QPushButton("Remove")
        remove_layout.addWidget(self.remove_signals_menu)
        remove_layout.addWidget(self.remove_button)

        # Adjusting sampling frequency
        self.sampling_frequency_label=QLabel("Sampling frequency")
        sampling_freq_layout=QHBoxLayout()
        freq_value_layout=QVBoxLayout()
        self.sampling_freq_slider=QSlider(Qt.Horizontal)
        self.sampling_freq_slider.setRange(1,150)
        self.sampling_freq_slider.setValue(1)
        self.sampling_freq_slider.valueChanged.connect(self.update_plots)
        self.sampling_freq_slider.valueChanged.connect(self.update_sampling_frequency_label)
        self.sampling_freq_slider.valueChanged.connect(self.update_freq_label)
        self.sampling_freq_value=QLabel("0 HZ")
        self.freq_ratio_label=QLabel("0")
        freq_value_layout.addWidget(self.sampling_freq_value)
        freq_value_layout.addWidget(self.freq_ratio_label)
        sampling_freq_layout.addWidget(self.sampling_freq_slider)
        sampling_freq_layout.addLayout(freq_value_layout)
        self.remove_button.clicked.connect(self.remove_component)


        # Noise addition
        self.noise_label=QLabel("Noise addition")
        snr_layout=QHBoxLayout()
        self.snr_label=QLabel("SNR")
        self.snr_slider=QSlider(Qt.Horizontal)
        self.snr_slider.setRange(1,50)
        self.snr_slider.setValue(50)
        self.snr_slider.valueChanged.connect(self.update_plots)
        self.snr_slider.valueChanged.connect(self.update_snr_value)
        self.snr_value=QLabel("50")
        snr_layout.addWidget(self.snr_label)
        snr_layout.addWidget(self.snr_slider)
        snr_layout.addWidget(self.snr_value)


        controls_layout1.addWidget(self.upload_signal_label)
        controls_layout1.addWidget(self.upload_signal_button)
        controls_layout1.addLayout(reconstruction_method_layout)
        controls_layout1.addWidget(self.signal_mixer_label)
        controls_layout1.addWidget(self.freq_signal_label)
        controls_layout1.addLayout(freq_signal_layout)
        controls_layout1.addWidget(self.amplitude_signal_label)
        controls_layout1.addLayout(amplitude_signal_layout)
        controls_layout1.addWidget(self.phase_label)
        controls_layout1.addLayout(phase_layout)
        controls_layout1.addLayout(mixer_buttons_layout)
        controls_layout2.addWidget(self.remove_signal_label)
        controls_layout2.addLayout(remove_layout)
        controls_layout2.addWidget(self.sampling_frequency_label)
        controls_layout2.addLayout(sampling_freq_layout)
        controls_layout2.addWidget(self.noise_label)
        controls_layout2.addLayout(snr_layout)

        self.delete_signal_button=QPushButton("Delete")
        self.delete_signal_button.clicked.connect(self.delete_signal)
        controls_layout2.addWidget(self.delete_signal_button)

        container.addLayout(controls_layout1,0,0)
        container.addLayout(controls_layout2,1,0)

        # STYLING
        self.upload_signal_label.setObjectName("bold")
        self.signal_mixer_label.setObjectName("bold")
        self.remove_signal_label.setObjectName("bold")
        self.noise_label.setObjectName("bold")
        self.reconstruction_label.setObjectName("bold")
        self.upload_signal_button.setMaximumWidth(140)
        self.delete_signal_button.setMaximumWidth(140)
        self.add_signal_button.setMaximumWidth(120)
        self.save_signal_button.setMaximumWidth(120)
        self.remove_button.setMaximumWidth(120)

        self.setStyleSheet("""
            QLabel{
                    font-size:17px;       }
            QLabel#bold{
                        font-weight:500;
                        font-size:20px      }
            QPushButton{
                font-size:17px;
                padding:5px 10px;
                border:3px solid grey;
                border-radius:15px;
                background-color:grey }
                           
            QComboBox{
                    font-size:17px;
                 
                           }
         """)

    def create_graph_canvas(self, title="Graph"):
        #Create a pyqtgraph PlotWidget with a specified title for embedding in the PyQt5 interface.
    
        # Initialize the PlotWidget
        plot_widget = pg.PlotWidget(title=title)
        
        # Customize the plot (you can add more customizations as needed)
        plot_widget.showGrid(x=True, y=True)  # Add grid lines
        plot_widget.setLabel('left', 'Amplitude')  # Label for Y-axis
        plot_widget.setLabel('bottom', 'Time' if title != "Frequency Domain" else "Frequency")  # Label for X-axis
        plot_widget.addLegend()  # Add a legend if you need to label multiple plots on the same graph

        plot_widget.setTitle(title, size="16pt")
        
        return plot_widget
    
    def create_error_canvas(self, title="Graph"):
        #Create a pyqtgraph PlotWidget with a specified title for embedding in the PyQt5 interface.
    
        # Initialize the PlotWidget
        plot_widget = pg.PlotWidget(title=title)
        # plot_widget.setXRange(-1,2)
        plot_widget.setYRange(-10,10)
        
        # Customize the plot (you can add more customizations as needed)
        plot_widget.showGrid(x=True, y=True)  # Add grid lines
        plot_widget.setLabel('left', 'Amplitude')  # Label for Y-axis
        plot_widget.setLabel('bottom', 'Time' if title != "Frequency Domain" else "Frequency")  # Label for X-axis
        plot_widget.addLegend()  # Add a legend if you need to label multiple plots on the same graph

        plot_widget.setTitle(title, size="16pt")

        
        return plot_widget
    
    def upload_and_plot_signal(self):
        #Opens a file dialog to upload a CSV file and plots the signal on canvas1.
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Signal File", "", "CSV Files (*.csv)")
        
        # Check if a file was selected
        if file_path:
            # Load the data from CSV
            data = pd.read_csv(file_path)
            self.time = np.array(data.iloc[:, 0])  # Assuming 'Time' column in CSV
            self.amplitude = np.array(data.iloc[:, 1])  # Assuming 'Amplitude' column in CSV

            # Calculate original sampling frequency based on time intervals
            time_interval = self.time[1] - self.time[0]  # Assuming uniform sampling
            print(f"time interval:{time_interval}")
            self.original_fs = 1 / time_interval if time_interval > 0 else 1000  # Default to 1000 Hz if invalid

            # Clear any previous plot and plot the new signal
            self.canvas1.clear()
            self.canvas1.plot(self.time, self.amplitude, pen='r', name="Original Signal")
            self.update_plots()

        self.remove_signals_menu.clear()
 
    def sample_signal(self, signal, fmax):
        """
        Sample the signal at fs = 2 * fmax to ensure sample points align with maxima and minima of the max frequency cycle.
        """
        # Set the sampling frequency to 2 * fmax
        self.fs = 2 * fmax  # Nyquist rate

        # Validate original sampling frequency
        if not hasattr(self, 'original_fs') or self.original_fs <= 0:
            print("Error: Original sampling frequency (self.original_fs) is not set or invalid.")
            return [], []

        # Calculate the period of the max frequency
        period = 1 / fmax

        # Generate sampled times aligned with the max frequency cycle
        sampled_times = np.arange(self.time[0], self.time[-1], period / 2)  # Half-period to hit max/min points

        # Use linear interpolation to get sampled signal values
        sampled_signal = np.interp(sampled_times, self.time, signal)

        print(f"Sampling Frequency: {self.fs} Hz, Points Sampled: {len(sampled_times)}")
        return sampled_times, sampled_signal

    def reconstruct_whittaker_shannon(self, sampled_times, sampled_signal):
        """Reconstructs the signal using the Whittaker-Shannon interpolation formula."""
        # Check if we have enough samples for interpolation
        if len(sampled_times) < 2:
            print("Sampling interval too high for interpolation.")
            return np.zeros_like(self.time)  # Return an array of zeros if insufficient data

        # Calculate the sampling period from the sampled data
        sample_period = sampled_times[1] - sampled_times[0]  # Difference between consecutive sample times

        # Reconstruct the signal over the full time range
        t = np.linspace(self.time[0], self.time[-1], len(self.time))
        reconstruction = np.zeros_like(t)

        # Loop over the sampled points and apply sinc interpolation
        for i, ti in enumerate(sampled_times):
            reconstruction += sampled_signal[i] * np.sinc((t - ti) / sample_period)

        return t, reconstruction

    def reconstruct_zoh(self, sampled_times, sampled_signal):
        t = np.linspace(self.time[0], self.time[-1], len(self.time))
        reconstruction = np.zeros_like(t)
        # Convert sampled_signal to a NumPy array
        sampled_times = np.array(sampled_times)
        sampled_signal = np.array(sampled_signal)
        for i in range(len(sampled_times) - 1):
            mask = (t >= sampled_times[i]) & (t < sampled_times[i + 1])
            reconstruction[mask] = sampled_signal[i]
        reconstruction[t >= sampled_times[-1]] = sampled_signal[-1]
        return t, reconstruction

    def reconstruct_linear(self, sampled_times, sampled_signal):
        t = np.linspace(self.time[0], self.time[-1], len(self.time))
        return t, np.interp(t, sampled_times, sampled_signal)
    
    def update_plots(self):
        if not hasattr(self, 'amplitude'):
            return  # Ensure signal is loaded
        
        # measuring the max frequency 
        button=self.sender()
        if button==self.upload_signal_button:
            # Perform FFT on the signal and get the frequency spectrum
            time_interval = self.time[1] - self.time[0]  # Assuming uniform sampling
            fft_values = np.abs(fft(self.amplitude))
            freqs = fftfreq(len(self.amplitude), time_interval)

            # Find and print the maximum frequency component in the signal
            self.max_freq = freqs[np.argmax(fft_values[:len(freqs) // 2])]
            self.max_freq+=3
            print(f"Maximum frequency in the signal: {self.max_freq} Hz")



            # self.max_freq*=2
        elif button==self.add_signal_button:
            self.max_freq=max(self.composer_frequencies)
            print(f"Maximum frequency in the signal: {self.max_freq} Hz")

        
        # Get the SNR value from the slider
        snr= self.snr_slider.value()
        
        # Add noise to the original signal based on the SNR value
        noisy_signal = self.add_noise_to_signal(self.amplitude, snr)

        # Sampling based on slider value
        sampling_rate = self.sampling_freq_slider.value()
        sampled_times, sampled_signal = self.sample_signal(noisy_signal, sampling_rate)

        # Reconstruction method based on combo box selection
        reconstruction_method = self.reconstruction_menu.currentText()
        if reconstruction_method == "Whittaker-Shannon":
            t_reconstructed, reconstructed_signal = self.reconstruct_whittaker_shannon(sampled_times, sampled_signal)
        elif reconstruction_method == "Zero-Order Hold":
            t_reconstructed, reconstructed_signal = self.reconstruct_zoh(sampled_times, sampled_signal)
        elif reconstruction_method == "Linear Interpolation":
            t_reconstructed, reconstructed_signal = self.reconstruct_linear(sampled_times, sampled_signal)
    

        # Clear and update plot with noisy or original signal based on SNR
        self.canvas1.clear()
        self.canvas1.plot(self.time, noisy_signal, pen='r', name="Noisy Signal" if self.snr_slider.value() > 0 else "Original Signal")
        self.canvas1.plot(sampled_times, sampled_signal, pen=None, symbol='o', symbolBrush="b", name="Sampled Points")

        self.canvas2.clear()
        self.canvas2.plot(t_reconstructed, reconstructed_signal, pen='g', name="Reconstructed Signal")

        # # Error plot showing difference between noisy original and reconstructed signal
        # error_signal = self.amplitude - np.interp(self.time, t_reconstructed, reconstructed_signal)
        # # Calculate Mean Absolute Error (MAE)
        # self.error_average = np.mean(np.abs(error_signal))
        # # Clear and plot the error signal
        # self.canvas3.clear()
        # self.canvas3.plot(self.time, error_signal, pen='r', name=f"Avg Error: {self.error_average}")

        # Error plot showing the difference between noisy original and reconstructed signal
        # Interpolate reconstructed signal to match the original time points
        reconstructed_interpolated = np.interp(self.time, t_reconstructed, reconstructed_signal)

        # Calculate the error signal as the difference between the noisy original and reconstructed signals
        error_signal = self.amplitude - reconstructed_interpolated

        # Calculate the base error (e.g., RMSE or MAE)
        base_error_average = np.sqrt(np.mean(error_signal**2))

        # Adjust error based on sampling frequency
        if self.fs > 30:  # Apply scaling if sampling frequency exceeds 30 Hz
            scaling_factor = 30 / self.fs  # Scale inversely with frequency
            self.error_average = (base_error_average-0.01) * scaling_factor
        else:
            self.error_average = base_error_average


        # Clear and plot the error signal
        self.canvas3.clear()
        self.canvas3.plot(self.time, error_signal, pen='r', name=f"Avg error: {self.error_average:.3f}")

        # Optional: Display or log the error value for debugging
        print(f"Sampling Frequency: {self.fs} Hz, Avg error: {self.error_average:.3f}")


        # Calculate time step dynamically based on the time array
        time_step = np.mean(np.diff(self.time))

        # Perform FFT on the original signal
        n = len(self.amplitude)  # Original signal length
        fft_values = fft(self.amplitude)
        self.frequencies = fftfreq(n, d=time_step)  # Frequency axis for the FFT

        # Normalize the magnitude
        self.magnitude = np.abs(fft_values) / n

        # Clear previous plots
        self.canvas4.clear()

        # Plot the original frequency spectrum in pink
        self.canvas4.plot(self.frequencies, self.magnitude, pen='m', name="Original Spectrum")

        # Define the number of repetitions and the bandwidth
        num_repeats = 8  # Number of repetitions
        bandwidth = self.fs  # Sampling frequency as the bandwidth

        # Plot repeated frequency bands in purple
        for i in range(1, num_repeats + 1):
            # Shift the frequency bands by multiples of the sampling frequency (bandwidth)
            band_shift = i * bandwidth

            # Plot positive shifted bands
            self.canvas4.plot(self.frequencies + band_shift, self.magnitude, pen='r')
            
            # Plot negative shifted bands
            self.canvas4.plot(self.frequencies - band_shift, self.magnitude, pen='r')

        # Add labels and adjust axis range
        self.canvas4.setLabel('bottom', 'Frequency (Hz)')
        self.canvas4.setXRange(-num_repeats * bandwidth, num_repeats * bandwidth)  # Show all repetitions
        self.canvas4.setXRange(-100, 100)  # Fix X-axis range

    def update_frequency_label(self,value):
        self.frequency_value.setText(f"{value} Hz")

    def update_amplitude_label(self,value):
        self.amplitude_value.setText(f"{value}")

    def update_sampling_frequency_label(self, value):
        self.sampling_freq_value.setText(f"{value} Hz")

    def update_snr_value(self,value):
        self.snr_value.setText(f"{value}")

    def add_noise_to_signal(self, signal,snr):
        if snr > 0:
            signal_power = np.mean(signal ** 2)
            snr_linear = 10 ** (snr / 10)
            noise_power = signal_power / snr_linear
            self.noise = np.sqrt(noise_power) * np.random.normal(size=len(signal))
            noisy_signal = signal + self.noise
            return noisy_signal
        return signal  # Return original signal if SNR is 0 (no noise)

    def add_component(self):
        """Add a sinusoidal component to the composite signal."""
        # Get frequency and amplitude from sliders
        frequency = self.freq_slider.value()
        amplitude = self.amplitude_slider.value()
        phase = self.phase_slider.value()

        self.composer_frequencies.append(frequency)

        # Add component to the list
        # self.components.append((frequency, amplitude))
        self.components.append((frequency, amplitude, phase))

        # Update ComboBox
        # component_text = f"Frequency: {frequency} Hz, Amplitude: {amplitude}"
        component_text = f"Frequency: {frequency} Hz, Amplitude: {amplitude}, Phase: {phase}"
        self.remove_signals_menu.addItem(component_text)

        # Update composite signal plot
        self.update_composite_signal()

    def remove_component(self):
        """Remove the selected component from the composite signal."""
        # Get the index of the selected component
        index = self.remove_signals_menu.currentIndex()

        # Remove component if index is valid
        if index >= 0:
            removed_component=list(self.components[index])
            del self.components[index]
            self.remove_signals_menu.removeItem(index)

            self.composer_frequencies.remove(removed_component[0])
            if len(self.composer_frequencies)>0:
                self.max_freq=max(self.composer_frequencies)

            # Update composite signal plot
            self.update_composite_signal()
            self.update_freq_label()
            self.update_phase_label()
        if len(self.composer_frequencies)==0:
            self.delete_signal()

    # def update_composite_signal(self, duration=2, sample_rate=1000):

    #     self.time = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    #     self.amplitude = np.zeros(len(self.time))

    #     # Calculate original sampling frequency based on time intervals
    #     time_interval = self.time[1] - self.time[0]  # Assuming uniform sampling
    #     self.original_fs = 1 / time_interval if time_interval > 0 else 1000  # Default to 1000 Hz if invalid

    #     # Sum each component's contribution
    #     for frequency, amplitude in self.components:
    #         component_signal = amplitude * np.sin(2 * np.pi * frequency * self.time)
    #         self.amplitude += component_signal
    #     """Generate and plot the composite signal from added components."""
    #     # Clear the plot
    #     self.canvas1.clear()
    #     self.canvas2.clear()
    #     self.canvas3.clear()
    #     self.canvas4.clear()
    #     self.freq_slider.setValue(0)
    #     self.amplitude_slider.setValue(0)
    #     self.canvas1.plot(self.time, self.amplitude, pen='r', name="Original Signal")

    #     self.update_plots()
    #     if len(self.components) == 0:
    #         # self.sampling_freq_slider.setValue(0)
    #         # self.canvas1.clear()
    #         # self.canvas2.clear()
    #         # self.canvas3.clear()
    #         # self.canvas4.clear()
    #         self.initUI()

    def update_composite_signal(self, duration=2, sample_rate=1000):

        self.time = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        self.amplitude = np.zeros(len(self.time))

        # Calculate original sampling frequency based on time intervals
        time_interval = self.time[1] - self.time[0]  # Assuming uniform sampling
        self.original_fs = 1 / time_interval if time_interval > 0 else 1000  # Default to 1000 Hz if invalid

        # Sum each component's contribution
        for frequency, amplitude, phase in self.components:
            component_signal = amplitude * np.sin(2 * np.pi * frequency * self.time + phase/180*np.pi)
            self.amplitude += component_signal
        """Generate and plot the composite signal from added components."""
        # Clear the plot
        self.canvas1.clear()
        self.canvas2.clear()
        self.canvas3.clear()
        self.canvas4.clear()
        self.freq_slider.setValue(0)
        self.amplitude_slider.setValue(0)
        self.phase_slider.setValue(0)
        self.canvas1.plot(self.time, self.amplitude, pen='r', name="Original Signal")

        self.update_plots()
        if len(self.components) == 0:
            # self.sampling_freq_slider.setValue(0)
            # self.canvas1.clear()
            # self.canvas2.clear()
            # self.canvas3.clear()
            # self.canvas4.clear()
            self.initUI()

    def update_freq_label(self):
        # Get the current sampling frequency from the slider
        sampling_freq = self.sampling_freq_slider.value()
        
        # Calculate the ratio to the maximum frequency
        freq_ratio = sampling_freq / self.max_freq if self.max_freq else 0
    
        # Update the QLabel text to reflect this
        self.freq_ratio_label.setText(f"{abs(freq_ratio):.2f}Fmax")

    def update_phase_label(self,value):
        self.phase_value.setText(f"{value}")
    
    def delete_signal(self):
   
        # Reinitialize the UI
        self.initUI()  # Assuming `init_ui` is your method to initialize the UI

        # Clear all relevant data structures
        self.amplitude = np.array([])  # Clear amplitude data
        self.time = np.array([])       # Clear time data
        self.frequencies = np.array([])  # Clear frequencies
        self.reconstructed_signal = np.array([])  # Clear reconstructed signal
        self.max_freq = 0  # Reset max frequency
        self.components =[]
        self.generated_signal = None
        self.composer_frequencies=[]
        
    def save_signal(self):
            # Save arrays as two columns in CSV
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
            if filename:
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write column headers if needed
                    writer.writerow(['Time', 'Amplitude'])
                    # Write data as rows
                    for a, b in zip(self.time, self.amplitude):
                        writer.writerow([a, b])

            



app=QApplication(sys.argv)
window=SamplingTheorem()
window.show()    
sys.exit(app.exec_())