from pathlib import Path

from app.services.background_removal_service import (
    BackgroundRemovalService,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".jfif",
}


def main():
    """
    Real batch-processing diagnostic test.

    Processes every supported image in new_testing_images
    through the actual BackgroundRemovalService.
    """

    print("=" * 70)
    print("BATCH BACKGROUND REMOVAL TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------

    images_folder = Path("new_testing_images")
    output_folder = Path("batch_test_outputs")

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # VALIDATE INPUT DIRECTORY
    # ---------------------------------------------------------

    if not images_folder.exists():
        print()
        print(f"❌ Input folder not found: {images_folder}")
        return

    if not images_folder.is_dir():
        print()
        print(f"❌ Input path is not a directory: {images_folder}")
        return

    # ---------------------------------------------------------
    # FIND IMAGES
    # ---------------------------------------------------------

    image_files = sorted(
        [
            file
            for file in images_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    print()
    print(f"Input folder : {images_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Images found : {len(image_files)}")
    print()

    if not image_files:
        print("⚠️ No supported images found.")
        return

    # ---------------------------------------------------------
    # LOAD SERVICE ONCE
    # ---------------------------------------------------------

    print("Initializing BackgroundRemovalService...")
    print()

    try:
        service = BackgroundRemovalService()
    except Exception as e:
        print("❌ Failed to initialize BackgroundRemovalService.")
        print(e)
        return

    # ---------------------------------------------------------
    # PROCESS IMAGES
    # ---------------------------------------------------------

    successful = []
    failed = []

    for index, image_file in enumerate(
        image_files,
        start=1,
    ):

        print("=" * 70)
        print(
            f"[{index}/{len(image_files)}] "
            f"Processing: {image_file.name}"
        )
        print("=" * 70)

        try:

            output_filename = (
                f"{image_file.stem}_output.png"
            )

            saved_path = service.remove_background(
                image_path=str(image_file),
                output_filename=output_filename,
            )

            print()
            print("✅ SUCCESS")
            print(f"Input : {image_file}")
            print(f"Output: {saved_path}")
            print()

            successful.append(image_file.name)

        except Exception as e:

            print()
            print("❌ FAILED")
            print(f"Image: {image_file.name}")
            print(f"Error: {e}")
            print()

            failed.append(
                (
                    image_file.name,
                    str(e),
                )
            )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)

    print()
    print(f"Total images : {len(image_files)}")
    print(f"Successful   : {len(successful)}")
    print(f"Failed       : {len(failed)}")
    print()

    if successful:
        print("-" * 70)
        print("SUCCESSFUL IMAGES")
        print("-" * 70)

        for name in successful:
            print(f"✓ {name}")

        print()

    if failed:
        print("-" * 70)
        print("FAILED IMAGES")
        print("-" * 70)

        for name, error in failed:
            print(f"✗ {name}")
            print(f"  {error}")

        print()

    print("=" * 70)

    if not failed:
        print("🎉 ALL IMAGES PROCESSED SUCCESSFULLY.")
    else:
        print("⚠️ BATCH COMPLETED WITH FAILURES.")

    print("=" * 70)


if __name__ == "__main__":
    main()