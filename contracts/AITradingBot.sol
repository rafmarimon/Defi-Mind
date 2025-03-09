// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
}

contract AITradingBot {
    address public owner;
    IUniswapV2Router public uniswapRouter;

    constructor(address _router) {
        owner = msg.sender;
        uniswapRouter = IUniswapV2Router(_router);
    }

    function executeTrade(
        address tokenIn,
        address tokenOut,
        uint amountIn,
        uint amountOutMin
    ) public {
        require(msg.sender == owner, "Only owner can trade");

        // Declare and initialize the path array properly
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;

        // Approve Uniswap Router to spend tokens
        IERC20(tokenIn).approve(address(uniswapRouter), amountIn);

        // Execute token swap
        uniswapRouter.swapExactTokensForTokens(
            amountIn, 
            amountOutMin, 
            path, 
            address(this), 
            block.timestamp + 300
        );
    }
}
