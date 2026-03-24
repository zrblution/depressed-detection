"""Prediction explanation helpers."""

from __future__ import annotations


EXPLANATION_PROMPT = """你是一个心理健康评估助手。以下是系统从用户社交媒体帖子中识别出的抑郁证据。

用户预测结果: {prediction_label}
预测置信度: {confidence:.2f}
主导模式: {dominant_channel}

证据帖子（按重要性排序）:
{evidence_list}

请基于以上证据生成一段客观、专业、不过度诊断的解释，控制在 200 字以内。
"""


def _template_explanation(prediction: dict) -> str:
    if prediction["label"] == 0:
        return (
            f"用户 {prediction['user_id']} 未检测到显著抑郁信号。"
            f"模型抑郁概率为 {prediction['depressed_logit']:.1%}，"
            f"当前最强通道为 {prediction['dominant_channel']}。"
        )
    evidence_summary = "; ".join(
        f"[{post_id}]={score:.2f}" for post_id, score in zip(prediction["evidence_post_ids"], prediction["evidence_scores"])
    )
    return (
        f"用户 {prediction['user_id']} 被判定为 depressed，抑郁概率 {prediction['depressed_logit']:.1%}。"
        f"主导通道为 {prediction['dominant_channel']}，关键证据为 {evidence_summary}。"
    )


def generate_explanation(prediction: dict, llm_client=None, *, model_name: str | None = None) -> str:
    evidence_list = ""
    for idx, (post_id, text, score) in enumerate(
        zip(prediction.get("evidence_post_ids", []), prediction.get("evidence_texts", []), prediction.get("evidence_scores", []))
    ):
        evidence_list += f"{idx + 1}. [{post_id}] ({score:.2f}) {text[:120]}\n"
    if llm_client is None or model_name is None:
        return _template_explanation(prediction)
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": EXPLANATION_PROMPT.format(
                    prediction_label="depressed" if prediction["label"] == 1 else "non_depressed",
                    confidence=prediction["depressed_logit"],
                    dominant_channel=prediction["dominant_channel"],
                    evidence_list=evidence_list,
                ),
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content

