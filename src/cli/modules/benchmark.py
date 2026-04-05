import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoTokenizer


class BenchmarkDB:
    """Manages OpenArc benchmark database operations."""
    
    def __init__(self, db_file: Optional[Path] = None):
        """
        Initialize BenchmarkDB with a database file path.
        
        Args:
            db_file: Path to the database file. If None, defaults to openarc_bench.db in project root.
        """
        if db_file is None:
            project_root = Path(__file__).parent.parent.parent.parent
            db_file = project_root / "openarc_bench.db"
        
        self.db_file = Path(db_file)
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize benchmark database and create table if it doesn't exist."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                bench_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                depth_tokens INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL,
                max_tokens INTEGER NOT NULL,
                run_number INTEGER NOT NULL,
                ttft_s TEXT,
                tpot_ms TEXT,
                prefill_throughput_tokens_s TEXT,
                decode_throughput_tokens_s TEXT,
                decode_duration_s TEXT,
                input_token_count TEXT,
                new_token_count TEXT,
                total_token_count TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        self._ensure_depth_column()

    def _ensure_depth_column(self) -> None:
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(benchmark_results)")
        cols = {row[1] for row in cursor.fetchall()}
        if "depth_tokens" not in cols:
            cursor.execute(
                "ALTER TABLE benchmark_results ADD COLUMN depth_tokens INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()
        conn.close()

    def save_result(self, model_name: str, result: Dict[str, Any], run_id: str) -> None:
        """
        Save a single benchmark result to the database.
        
        Args:
            model_name: Name of the model being benchmarked.
            result: Dictionary containing benchmark results with keys:
                    'd', 'p', 'n', 'run', 'ttft', 'tpot', 'prefill_throughput',
                    'decode_throughput', 'decode_duration', 'input_token',
                    'new_token', 'total_token'
            run_id: Unique identifier for the benchmark run.
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO benchmark_results (
                run_id, timestamp, model_name, depth_tokens, input_tokens, max_tokens, run_number,
                ttft_s, tpot_ms, prefill_throughput_tokens_s, decode_throughput_tokens_s,
                decode_duration_s, input_token_count, new_token_count, total_token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now().isoformat(),
            model_name,
            int(result.get("d", 0)),
            result['p'],
            result['n'],
            result['run'],
            str(result['ttft']),
            str(result['tpot']),
            str(result['prefill_throughput']),
            str(result['decode_throughput']),
            str(result['decode_duration']),
            result['input_token'],
            result['new_token'],
            result['total_token']
        ))
        
        conn.commit()
        conn.close()


class OpenArcBenchmarks:
    """Utilities for OpenArc benchmarking operations."""
    
    @staticmethod
    def random_input_ids(model_path: str, num_tokens: int, *, depth: int = 0) -> list:
        """
        Generate random input tokens for benchmarking.
        Follows llama.cpp approach.
        https://github.com/ggml-org/llama.cpp/blob/683fa6ba/tools/llama-bench/llama-bench.cpp#L1922

        When ``depth`` > 0, that many tokens are sampled first as synthetic prior
        context; ``num_tokens`` additional tokens follow (the swept prompt segment).
        
        Args:
            model_path: Path to the model.
            num_tokens: Number of prompt tokens after the optional prefix.
            depth: Random vocab tokens prepended as fake prior context (default 0).
            
        Returns:
            List of random token IDs of length ``depth + num_tokens``.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        vocab_size = len(tokenizer)
        
        special_token_ids = set(tokenizer.all_special_ids)
        valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]

        def sample(n: int) -> list:
            return [random.choice(valid_token_ids) for _ in range(n)]

        return sample(depth) + sample(num_tokens)


# Example usage:
# if __name__ == "__main__":
#     model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral/Impish_Nemo_12B-int4_asym-awq-ov"
#     num_tokens = 512
#     
#     input_ids = OpenArcBenchmarks.random_input_ids(model_path, num_tokens)
#     print(f"Generated {len(input_ids)} random tokens")
#     print(f"Sample tokens: {input_ids[:10]}")
    
