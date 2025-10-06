import asyncio


async def print_even_numbers():
    for i in range(0, 11, 2):
        print(i)
        await asyncio.sleep(1)
    
async def print_odd_numbers():
    for i in range(1, 11, 2):
        print(i)
        await asyncio.sleep(1)

async def main():
    await asyncio.gather(print_even_numbers(), print_odd_numbers())


if __name__ == "__main__":
    asyncio.run(main())

