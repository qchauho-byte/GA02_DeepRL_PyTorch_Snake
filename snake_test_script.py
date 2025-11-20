from game_environment import Snake


def main():
    """
    Let a human player control the snake with keyboard input.
    This file is only for quick testing and is not used in training.
    """

    # Create the environment
    # Snake here only takes board_size and frames (one game only)
    env = Snake(board_size=10, frames=2)

    # Reset the environment
    state = env.reset()

    done = False

    while not done:
        try:
            # Get action from user
            # -1 = turn left, 0 = go straight, 1 = turn right
            action = int(input("Enter action [-1, 0, 1] : "))
        except ValueError:
            print("Please type -1, 0 or 1.")
            continue

        # Step the environment
        step_result = env.step(action)

        # Handle different possible return formats
        if len(step_result) == 3:
            state, reward, done = step_result
            info = {}
        elif len(step_result) == 4:
            state, reward, done, info = step_result
        else:
            state, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        # Print the updated game
        env.print_game()
        print(f"Reward from this step: {reward}")

    print("Game over. Final reward:", reward)


if __name__ == "__main__":
    main()
