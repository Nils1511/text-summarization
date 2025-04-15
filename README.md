# Modular Text Summarization System

A well-structured, modular Python application for text summarization using self-hosted transformer-based models. This project provides a complete pipeline for loading data, performing summarization, evaluating results, and analyzing the performance.

## Features

- **Modular Design**: Clean, class-based architecture with separation of concerns
- **Self-hosted Models**: Uses local transformer models from Hugging Face (no API dependencies)
- **Comprehensive Evaluation**: ROUGE metrics and detailed error analysis
- **Visualization**: Results visualization for better understanding of model performance
- **Hyperparameter Tuning**: Simple mechanism for tuning generation parameters
- **Configurable**: YAML-based configuration for all components

## Architecture

The project follows a modular, object-oriented design with the following components:

- **`data_loader.py`**: Handles dataset loading, preprocessing, and batching
- **`model.py`**: Manages model loading, configuration, and summary generation
- **`evaluator.py`**: Performs evaluation and error analysis of generated summaries
- **`dialogue_processor.py`**: Specialized component for processing and analyzing dialogue data
- **`pipeline.py`**: Orchestrates the full summarization pipeline
- **`main.py`**: Command-line interface for running different modes

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/text-summarization.git
   cd text-summarization
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Configuration

The system is configured through `config.yaml`. The default configuration uses the BART model and the SAMSum dataset, but you can customize this according to your needs.

### Running the Pipeline

#### Evaluation Mode

Evaluate the summarization model on a dataset split:

```bash
python main.py --mode evaluate --split test --samples 100 --output_dir results
```

#### Hyperparameter Tuning

Tune the summarization hyperparameters:

```bash
python main.py --mode tune --split validation --samples 200 --output_dir tuning_results
```

#### Summarization Mode

Summarize a specific text:

```bash
python main.py --mode summarize --text "Your text to summarize goes here"
```

Or pipe text into the script:

```bash
cat document.txt | python main.py --mode summarize
```

### Example Workflow

1. Configure the system in `config.yaml`
2. Run evaluation on a small sample to verify everything works:
   ```
   python main.py --mode evaluate --split test --samples 20
   ```
3. Tune hyperparameters to find optimal settings:
   ```
   python main.py --mode tune --split validation --samples 100
   ```
4. Update config with best parameters and run full evaluation:
   ```
   python main.py --mode evaluate --split test
   ```
5. Use the model for summarizing new texts:
   ```
   python main.py --mode summarize --text "Your text here"
   ```

## Dataset

The default configuration uses the [SAMSum dataset](https://huggingface.co/datasets/samsum), which contains dialogues with their corresponding summaries. This dataset is automatically downloaded through the Hugging Face `datasets` library.

The dataset contains:
- 14,732 training examples
- 818 validation examples
- 819 test examples

## Models

By default, this project uses the BART model (`philschmid/bart-large-cnn-samsum`) from Hugging Face, but you can easily switch to other models like T5, Pegasus, or DistilBART by changing the configuration.

## Dialogue Processing

The system includes specialized dialogue processing capabilities through the `DialogueProcessor` class, which:

- Extracts speakers from conversation text
- Identifies key actions and topics for each speaker
- Creates structured representation of dialogues
- Provides contextual information to enhance summarization
- Performs dialogue-specific analysis for better understanding of conversation dynamics

## Evaluation Metrics

The system evaluates summaries using:
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)

Additionally, it performs comprehensive error analysis to identify strengths and weaknesses of the summarization approach.

## Design Choices and Approach

### Modular Class Structure

The project follows a strict modular design to ensure:
- **Maintainability**: Each component has a single responsibility
- **Extensibility**: Easy to add new models, datasets, or evaluation metrics
- **Testability**: Components can be tested in isolation

### Error Analysis

Beyond simple metrics, the evaluator performs detailed error analysis:
- Length analysis (relationship between input/output lengths)
- Content analysis (missed or hallucinated content)
- Challenging cases identification (where the model performs poorly)
- Statistical analysis of performance metrics

### Logging and Configuration

The system uses:
- Comprehensive logging for tracking execution and debugging
- YAML-based configuration for easy modification of parameters
- Command-line interface for flexible usage

## Limitations and Future Work

### Current Limitations

- No fine-tuning implementation for customizing models
- Limited to transformer-based sequence-to-sequence models
- Evaluation focused on reference-based metrics (ROUGE)

### Future Improvements

- Add model fine-tuning capabilities
- Implement reference-free evaluation metrics
- Add more advanced preprocessing for dialogues
- Create a simple web interface for interactive summarization
- Support for extractive summarization methods
- Multi-document summarization support

## Results and Insights

For detailed findings and analysis from running this system, please refer to the [INSIGHTS.md](INSIGHTS.md) document.

## License

MIT License