from data_manager import DataStream, DataSet, DataSource
from processing import DataProcessor
from data_types import RawData, ProcessedData, MessageType
from visualization import LiveVisualizer, RotationVisualizer, LegVisualizer, AngleVisualizer
import time

def main():
    data_source = None
    visualizer = None
    rotation_viz = None
    leg_viz = None
    angle_viz = None
    try:
        data_source = DataStream("/dev/ttyACM0")
        data_processor = DataProcessor()
        visualizer = LiveVisualizer()
        rotation_viz = RotationVisualizer()
        leg_viz = LegVisualizer()
                
        input("Collecting data for regression and setting extension point, press enter to continue...")
        i = 0

        while i < 200:
            message = data_source.sample()
            print(message)

            if message is None or message.msg_type != MessageType.DATA:
                continue
                
            data = message.data
            if message.msg_type == MessageType.DATA:
                if not (data.calibration >= 2).all():
                    print("Sensors not fully calibrated, skipping sample")
                    continue
                
                data_processor.add_to_regression(data)
                rotation_viz.update(data)
                visualizer.update(data)
                leg_viz.update(data)
                if i % 100 == 0:
                    print(f"Collected {i} samples")
                i += 1

        data_processor.set_extension_point(data_processor.regression_samples[0])
        data_processor.fit_regression()
            
        # Add regression line to visualization
        visualizer.set_regression_line(data_processor.regression_model)
        
        # Create angle visualizer after regression is complete
        angle_viz = AngleVisualizer()
        
        while True:
            message = data_source.sample()
            if message is None or message.msg_type != MessageType.DATA:
                continue
                
            raw_data = message.data
            if raw_data is None:
                continue
                
            processed_data = data_processor.process(raw_data)
            visualizer.update(raw_data)
            rotation_viz.update(raw_data)
            leg_viz.update(raw_data)
            angle_viz.update(processed_data)
            # print(f"Processed data: {processed_data}")
            
    finally:
        # Clean up
        if data_source:
            del data_source
        if visualizer:
            visualizer.close()
        if rotation_viz:
            rotation_viz.close()
        if leg_viz:
            leg_viz.close()
        if angle_viz:
            angle_viz.close()

if __name__ == "__main__":
    main()
    
    