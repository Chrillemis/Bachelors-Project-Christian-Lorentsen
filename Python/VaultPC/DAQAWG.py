
import nidaqmx as ni
from nidaqmx import stream_readers, stream_writers
import pandas as pd
import numpy as np


def read_chunk_from_csv(
        csv_file_name = "Wigner.csv",
        sr = 50_000, # Sampling rate in Hz
        column = 0 # Column to read from the CSV file
):
    '''
    Reads a chunk of data from a CSV file and returns it as a numpy array, that fits the DAQ.
    '''
    chunk_reader = pd.read_csv(csv_file_name, chunksize = sr/10, usecols = [column])
    while True:
        chunk = next(chunk_reader).to_numpy()[:,0]
        data = np.repeat(np.expand_dims(chunk, axis = 1), 2, axis =1).T
        yield np.ascontiguousarray(data)

def DAQ(
        sr = 50_000, #Sampling rate in Hz
        duration = 10, 
        data_file = 'test3', #name of the output file without extension
        column = 0, #Column to read from the CSV file
        csv_file_name = "test.csv", #name of the input CSV file

        input_mapping = ['Dev1/ai0', 'Dev1/ai1'], #Devices to read from
        output_mapping = ['Dev1/ao0', 'Dev1/ao1'], #Devices to write to
        ao_args = {'min_val': -10,
                'max_val': 10}, #Arguments for the output channels
        ai_args = {'min_val': -10, #Arguments for the input channels
                'max_val': 10,
                'terminal_config': ni.constants.TerminalConfiguration.RSE}

):
    '''
    Reads data from a CSV file in chunks and writes it to a DAQ device.
    The data is saved in a TDMS file, named {data_file}.tdms.
    '''
    nsamples = sr*duration

    #Column names in the TDMS file
    group_name = "SamplingRate" + str(sr) + "Time" + str(duration) + "s" + pd.read_csv(csv_file_name, nrows=0).columns[column]


    with ni.Task() as read_task, ni.Task() as write_task:
        for o in output_mapping:
            write_task.ao_channels.add_ao_voltage_chan(o, **ao_args)

        for i in input_mapping:
            read_task.ai_channels.add_ai_voltage_chan(i, **ai_args)

        for task in (read_task, write_task):
            task.timing.cfg_samp_clk_timing(rate=sr, source='OnboardClock', samps_per_chan=nsamples)
        

        read_task.in_stream.configure_logging(data_file + ".tdms", group_name = group_name)

        # reader = stream_readers.AnalogMultiChannelReader(read_task.in_stream)
        writer = stream_writers.AnalogMultiChannelWriter(write_task.out_stream)

        write_task.triggers.start_trigger.cfg_dig_edge_start_trig(read_task.triggers.start_trigger.term)
        writer.write_many_sample(next(read_chunk_from_csv(csv_file_name=csv_file_name, sr=sr, column = column)), timeout=0)

        write_task.start()
        read_task.start()

        read_task.read(nsamples)

import stochastic
from stochastic.processes.diffusion import *

def ContinousStocastichProcess(amplitude: float = 5, 
                               center: float = 7.5, 
                               samples_per_bunch: int = 5_000, 
                               processes = []
                               ):
    '''
    Generates a list of stochastic processes and returns them as a numpy array.
    The processes are scaled to fit the DAQ.
    '''
    values = np.zeros((len(processes), samples_per_bunch), dtype=np.float64)
    for j in range(len(processes)):
        try:
            samples = processes[j].sample(samples_per_bunch)[:samples_per_bunch]
            scaled = (samples - np.min(samples))
            scaled = amplitude*scaled/np.max(scaled) - amplitude/2 + center
            values[j] = scaled
        except Exception as e:
            print(e)
            
    return values


def processes(sr, duration, samples_per_frame):
    different_distributions = int(sr*duration/samples_per_frame)
    random_numbers = np.random.uniform(0, 1, different_distributions+1)
    random_numbers2 = np.random.uniform(0, 1, different_distributions+1)
    # # random_numbers3 = np.random.uniform(0, 1, different_distributions+1)
    # random_numbers = np.ones(different_distributions+1)*1/2
    # random_numbers2 = np.ones(different_distributions+1)*1/2
    for i in range(different_distributions+1):
        yield np.array([
                #Continuous time processes
                stochastic.processes.continuous.BesselProcess(int(random_numbers[i]*9 + 1)), #  Bessel process of order 1 to 10
                stochastic.processes.continuous.BrownianBridge(),
                stochastic.processes.continuous.BrownianExcursion(),
                stochastic.processes.continuous.BrownianMeander(),
                stochastic.processes.continuous.BrownianMotion(),
                stochastic.processes.continuous.CauchyProcess(), #
                stochastic.processes.continuous.FractionalBrownianMotion(random_numbers[i]*0.98 + 0.01), # Hurst parameter between 0.01 and 0.99
                stochastic.processes.continuous.GammaProcess(mean = 5, variance=2.5), #
                stochastic.processes.continuous.GeometricBrownianMotion(drift = 2*random_numbers[i], volatility= 1*random_numbers2[i]+0.5), #
                stochastic.processes.continuous.InverseGaussianProcess(), #
                stochastic.processes.continuous.MixedPoissonProcess(lambda: 100), #
                ## A multifractional Brownian motion generalizes a fractional Brownian motion 
                ## with a Hurst parameter which is a function of time.
                ## hurst oscillating 
                # stochastic.processes.continuous.MultifractionalBrownianMotion(hurst = lambda x: (np.sin((random_numbers[i]*18 + 2)*x)**2)*0.98+ 0.01), #er god nok(), men langsom #
                stochastic.processes.continuous.PoissonProcess(), #
                stochastic.processes.continuous.SquaredBesselProcess(int(random_numbers[i]*9 + 1)), #
                stochastic.processes.continuous.VarianceGammaProcess(drift = 2*random_numbers[i], variance = 0.01, scale = 100), #
                stochastic.processes.continuous.WienerProcess(),
                #Diffusion models
                DiffusionProcess(speed=10*random_numbers[i], mean=0, vol=10*random_numbers[i], volexp=0, t=1, rng=None),
                ConstantElasticityVarianceProcess(drift = 1, vol=10*random_numbers[i], volexp=0),
                CoxIngersollRossProcess(speed=10*random_numbers[i], mean=0, vol=10*random_numbers[i]),
                OrnsteinUhlenbeckProcess(speed=10*random_numbers[i], vol=10*random_numbers[i]),
                VasicekProcess(speed=10*random_numbers[i], mean=0, vol=10*random_numbers[i]),
                stochastic.processes.noise.GaussianNoise(),
                stochastic.processes.noise.FractionalGaussianNoise(random_numbers[i]*0.98 + 0.01),
                # Colored noise
                stochastic.processes.noise.BlueNoise(),
                stochastic.processes.noise.BrownianNoise(),
                stochastic.processes.noise.ColoredNoise(),
                stochastic.processes.noise.RedNoise(),
                stochastic.processes.noise.PinkNoise(),
                stochastic.processes.noise.VioletNoise(),
                stochastic.processes.noise.WhiteNoise()
            ])


def Data_gen(
            
            amplitude = 5,
            center = 7.5,
            samples_per_frame = 5_000,
            sr = 50_000,
            duration = 10,
            tail_len = 0,
            csv_file = "test.csv"
):

    """
    Generates a csv_file from a list of stochastic processes, with a given tail length.
    """


    old_samples_per_frame = samples_per_frame
    if tail_len > 0:
        samples_per_frame = samples_per_frame - tail_len
    values_list = []
    for seed in range(int(sr*duration/old_samples_per_frame)):

        values = ContinousStocastichProcess(samples_per_bunch=samples_per_frame, amplitude=amplitude, center = center, 
                                            processes=next(processes(sr = sr, duration = duration, samples_per_frame = old_samples_per_frame)))
        values_list.append(values)
        if tail_len > 0:
            tail = np.ones((len(processes), tail_len), dtype=np.float64) * center
            values_list.append(tail)

    values_combined = np.concatenate(values_list, axis = 1)

    process_arr = next(processes(sr = sr, duration = duration, samples_per_frame = old_samples_per_frame))
    nr_of_processes = len(process_arr)
    keys = np.zeros((nr_of_processes,), dtype='<U64')
    for i in range(nr_of_processes):
        keys[i] = process_arr[i].__doc__.split('.')[0]

    pd.DataFrame(values_combined.T, columns=keys).to_csv(csv_file, index=False)
    
def Long_OU_jump(amplitude = 5,
                center = 7.5,
                samples_per_bunch = 5_000,
                sr = 500_000,
                duration = 1000,
                tail_len = 0,
                csv_file = "LongOUJump.csv",
                NR_OF_JUMPS = 50 #per bunch
                ):
        
    if duration%100 != 0:
        print("Failed, duration must be divisible by 100")
    all_values = np.zeros((int(duration/100), 2*samples_per_bunch))#100*sr))
    keys = ["OUProcess" + str(i) for i in range(int(duration/100))]

    for i in range(int(duration/100)):
    
        values = []
        print("col", i)
        for _ in range(int(100*sr/samples_per_bunch)):
            process = OrnsteinUhlenbeckProcess(speed=5, vol=5)
            samples = process.sample(samples_per_bunch)[:samples_per_bunch]
            if NR_OF_JUMPS > 0:
                jump = np.repeat(
                    np.random.uniform(center - amplitude/2, center + amplitude/2, NR_OF_JUMPS), 
                    samples_per_bunch/NR_OF_JUMPS
                    )
                samples += jump
            scaled = (samples - np.min(samples))
            scaled = amplitude*scaled/np.max(scaled) - amplitude/2 + center

            values.append(scaled)

        all_values[i] = np.concatenate(values)
    pd.DataFrame(all_values.T, columns=keys).to_csv(csv_file, index=False)


def DAQ_one_col(csv_file, data_file_name, sr, duration):
    cols = pd.read_csv(csv_file, nrows=0).columns
    for i in range(len(cols)):
        print(f"Column {i}: {cols[i]}")
        try:
            DAQ(
            sr = sr, #Sampling rate in Hz
            duration = duration, 
            data_file = data_file_name, #name of the output file without extension
            column = i, #Column to read from the CSV file
            csv_file_name= csv_file
            )
        except Exception as e:
            print(f"Error with column {i}: {e}")
            continue