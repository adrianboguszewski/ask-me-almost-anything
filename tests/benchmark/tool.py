import asyncio
import logging as log
import time
from collections import defaultdict

import httpx


class BenchmarkTool:

    def __init__(self, url: str, num_users: int, request_length: int, iterations: int):
        self.url = url
        self.num_users = num_users
        self.request_length = request_length
        self.iterations = iterations

    async def send_request(self, question: str):
        async with httpx.AsyncClient() as client:
            try:
                start_time = time.perf_counter()
                # 400 is also ok as the output was generated but filtered out by the guard
                await client.post(self.url, json={"question": question}, timeout=None)
                end_time = time.perf_counter()

                return end_time - start_time, True
            except RuntimeError as e:
                return float("inf"), False

    async def simulate_user(self, length: int):
        question = "A" * length
        response_time, success = await self.send_request(question)
        return response_time, success

    async def run_benchmark(self):
        log.info(f"Testing with request length: {self.request_length} and {self.num_users} users")

        results = defaultdict(list)
        for i in range(self.iterations):
            log.info(f"Running iteration {i + 1} of {self.iterations}")
            tasks = [self.simulate_user(self.request_length) for _ in range(self.num_users)]
            start_time = time.perf_counter()
            responses = await asyncio.gather(*tasks)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            response_times = [r[0] for r in responses]
            successes = [r[1] for r in responses]

            avg_response_time = sum(response_times) / len(response_times)
            success_rate = sum(successes) / len(successes) * 100
            results["total_time"].append(total_time)
            results["response_time"].append(avg_response_time)
            results["success_rate"].append(success_rate)

            log.debug(f"Avg Response Time={avg_response_time:.2f}s, Success Rate={success_rate:.2f}%, Total Time={total_time:.2f}s")

        self.show_summary(results)

    @staticmethod
    def show_summary(results):
        avg_response_time = sum(results["response_time"]) / len(results["response_time"])
        total_time = sum(results["total_time"]) / len(results["total_time"])
        success_rate = sum(results["success_rate"]) / len(results["success_rate"])
        log.info("Summary:")
        log.info(f"Avg Response Time={avg_response_time:.2f}s, Success Rate={success_rate:.2f}%, Total Time={total_time:.2f}s")


if __name__ == "__main__":
    log.getLogger().setLevel(log.INFO)

    import argparse

    parser = argparse.ArgumentParser(description="Benchmark FastAPI Server Performance")
    parser.add_argument("--url", type=str, default="http://localhost:8000/answer", help="Server URL")
    parser.add_argument("--iterations", type=int, default=25, help="Number of test iterations")
    parser.add_argument("--num_users", type=int, default=1, help="Number of concurrent users")
    parser.add_argument("--request_length", type=int, default=50, help="Request length")
    args = parser.parse_args()

    benchmark = BenchmarkTool(url=args.url, num_users=args.num_users, request_length=args.request_length, iterations=args.iterations)

    asyncio.run(benchmark.run_benchmark())
