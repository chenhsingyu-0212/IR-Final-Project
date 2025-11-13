import json
import sys
from pathlib import Path

def calculate_time_stats(json_file_path):
    """
    計算JSON文件中時間相關的統計數據
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = data['results']
        summary = data['summary']

        # 計算詳細統計
        total_initial_time = sum(r['initial_time_sec'] for r in results)
        total_rag_time = sum(r['rag_time_sec'] for r in results)
        total_time = sum(r['total_time_sec'] for r in results)

        avg_initial_time = total_initial_time / len(results)
        avg_rag_time = total_rag_time / len(results)
        avg_total_time = total_time / len(results)

        # 計算檢索使用統計
        retrieval_count = sum(1 for r in results if r['used_retrieval'])
        retrieval_percentage = (retrieval_count / len(results)) * 100

        return {
            'file': Path(json_file_path).name,
            'total_samples': len(results),
            'total_initial_time_sec': total_initial_time,
            'total_rag_time_sec': total_rag_time,
            'total_time_sec': total_time,
            'avg_initial_time_sec': avg_initial_time,
            'avg_rag_time_sec': avg_rag_time,
            'avg_total_time_sec': avg_total_time,
            'retrieval_count': retrieval_count,
            'retrieval_percentage': retrieval_percentage,
            'summary_avg_time': summary.get('average_time_sec', 0)
        }

    except FileNotFoundError:
        print(f"錯誤：找不到文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"錯誤：JSON文件格式錯誤 {json_file_path}")
        return None
    except KeyError as e:
        print(f"錯誤：JSON文件缺少必要字段 {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("使用方法: python time_calculator.py <json_file1> [json_file2] ...")
        print("範例: python time_calculator.py exp1_with_scores.json exp2_with_scores.json")
        return

    json_files = sys.argv[1:]

    print("=" * 60)
    print("時間統計計算器")
    print("=" * 60)

    all_stats = []
    for json_file in json_files:
        stats = calculate_time_stats(json_file)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("沒有有效的數據進行計算")
        return

    # 顯示每個文件的統計
    for stats in all_stats:
        print(f"\n文件: {stats['file']}")
        print("-" * 40)
        print(f"樣本數量: {stats['total_samples']}")
        print(f"總初始時間: {stats['total_initial_time_sec']:.2f} 秒")
        print(f"總RAG時間: {stats['total_rag_time_sec']:.2f} 秒")
        print(f"總時間: {stats['total_time_sec']:.2f} 秒")
        print(f"平均初始時間: {stats['avg_initial_time_sec']:.2f} 秒")
        print(f"平均RAG時間: {stats['avg_rag_time_sec']:.2f} 秒")
        print(f"平均總時間: {stats['avg_total_time_sec']:.2f} 秒")
        print(f"使用檢索樣本數: {stats['retrieval_count']} ({stats['retrieval_percentage']:.1f}%)")
        print(f"摘要平均時間: {stats['summary_avg_time']:.2f} 秒")

    # 如果有多個文件，顯示比較
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("比較分析")
        print(f"{'='*60}")

        base_stats = all_stats[0]
        for stats in all_stats[1:]:
            print(f"\n{base_stats['file']} vs {stats['file']}:")
            time_diff = stats['avg_total_time_sec'] - base_stats['avg_total_time_sec']
            print(f"平均總時間差異: {time_diff:+.2f} 秒 ({time_diff/base_stats['avg_total_time_sec']*100:+.1f}%)")

            rag_diff = stats['avg_rag_time_sec'] - base_stats['avg_rag_time_sec']
            print(f"平均RAG時間差異: {rag_diff:+.2f} 秒")

            retrieval_diff = stats['retrieval_percentage'] - base_stats['retrieval_percentage']
            print(f"檢索使用率差異: {retrieval_diff:+.1f}%")

if __name__ == "__main__":
    main()