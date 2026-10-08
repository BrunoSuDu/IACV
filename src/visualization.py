from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def save_keypoint_visualization(
    image_color,
    keypoints,
    title,
    save_path,
    show=False,
):
    image_vis = cv2.drawKeypoints(
        image_color,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    image_vis = cv2.cvtColor(
        image_vis,
        cv2.COLOR_BGR2RGB,
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(image_vis)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close()


def save_match_visualization(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches,
    title,
    save_path,
    max_matches=100,
    show=False,
):
    matches = sorted(
        matches,
        key=lambda match: match.distance,
    )

    matches_to_draw = matches[:max_matches]

    image_vis = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    image_vis = cv2.cvtColor(
        image_vis,
        cv2.COLOR_BGR2RGB,
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(image_vis)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close()
