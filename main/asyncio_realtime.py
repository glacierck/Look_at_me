from asyncio import queues
import asyncio
from my_insightface.insightface.app.asyncio_tracker import AsyncioFaceAnalysis, COST_TIME


async def main():
    test = AsyncioFaceAnalysis(test_folder='test_04')
    video_2_detect_queue = asyncio.Queue(maxsize=200)
    detect_2_rec_queue = asyncio.Queue(maxsize=400)
    rec_2_show_queue = asyncio.Queue(maxsize=400)
    try:

        await asyncio.gather(
            test.video_read(results=video_2_detect_queue),
            test.image2detect(jobs=video_2_detect_queue, results=detect_2_rec_queue),
            test.detect2identify(jobs=detect_2_rec_queue, results=rec_2_show_queue),
            test.image_show(jobs=rec_2_show_queue)
        )
    except Exception as e:
        print(f'Exception occurs, error = {e}')
        raise e
    finally:
        print('all thread tasks done')
        print('ave_fps = ', test.show_times / COST_TIME['image_show'][0])
        # test.test_stop()


if __name__ == '__main__':
    asyncio.run(main())
