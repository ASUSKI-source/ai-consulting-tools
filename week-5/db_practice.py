"""Demonstration of all four CRUD operations using SQLAlchemy."""

from database import Analysis, SessionLocal, create_tables
from datetime import datetime

# Initialize the database tables.
create_tables()


# ==============================================================================
# SECTION 1: CREATE (Insert rows)
# ==============================================================================
print('--- SECTION 1: CREATE ---')

# Create a new session for database operations.
session = SessionLocal()

# Create the first Analysis object for AAPL (Apple stock).
analysis_aapl = Analysis(
    ticker='AAPL',
    asset_type='stock',
    company_name='Apple Inc.',
    current_price=195.25,
    rsi=62.5,
    sma_5=193.10,
    sma_20=190.45,
    momentum_5d=2.15,
    momentum_30d=5.80,
    market_cap=3.2e12,
    ai_analysis='Apple shows strong uptrend with RSI approaching overbought territory.',
    prompt_tokens=150,
    completion_tokens=180,
    estimated_cost=0.0035,
)

# Create the second Analysis object for TSLA (Tesla stock).
analysis_tsla = Analysis(
    ticker='TSLA',
    asset_type='stock',
    company_name='Tesla Inc.',
    current_price=242.80,
    rsi=58.3,
    sma_5=240.50,
    sma_20=238.75,
    momentum_5d=4.05,
    momentum_30d=8.12,
    market_cap=780e9,
    ai_analysis='Tesla demonstrates bullish momentum with support near 240 level.',
    prompt_tokens=155,
    completion_tokens=175,
    estimated_cost=0.0033,
)

# Create the third Analysis object for BTC (Bitcoin cryptocurrency).
analysis_btc = Analysis(
    ticker='BTC',
    asset_type='crypto',
    company_name='Bitcoin',
    current_price=52340.75,
    rsi=70.2,
    sma_5=51850.00,
    sma_20=50125.50,
    momentum_5d=2320.75,
    momentum_30d=8215.25,
    market_cap=1.04e12,
    ai_analysis='Bitcoin is consolidating after a strong rally; watch for breakout signals.',
    prompt_tokens=160,
    completion_tokens=185,
    estimated_cost=0.0036,
)

# Add all three Analysis objects to the session.
session.add(analysis_aapl)
session.add(analysis_tsla)
session.add(analysis_btc)

# Commit the transaction to persist all records to the database.
session.commit()
print('Saved 3 analyses to database.')

# Close the session to free up database resources.
session.close()


# ==============================================================================
# SECTION 2: READ (Query rows)
# ==============================================================================
print('\n--- SECTION 2: READ ---')

# Open a new session for read operations.
session = SessionLocal()

# Query 1: Retrieve ALL analyses from the database.
all_analyses = session.query(Analysis).all()
print(f'\nQuery 1 - All analyses:')
for analysis in all_analyses:
    print(f'  id={analysis.id}, ticker={analysis.ticker}, current_price={analysis.current_price}')

# Query 2: Retrieve only analyses where asset_type is 'stock'.
stock_analyses = session.query(Analysis).filter(Analysis.asset_type == 'stock').all()
print(f'\nQuery 2 - Stock analyses found: {len(stock_analyses)}')

# Query 3: Get the single most recent analysis using order_by and first().
most_recent = session.query(Analysis).order_by(Analysis.created_at.desc()).first()
print(f'\nQuery 3 - Most recent analysis:')
print(f'  ticker={most_recent.ticker}, created_at={most_recent.created_at}')

# Query 4: Get all analyses for the specific ticker 'AAPL' using filter().
aapl_analyses = session.query(Analysis).filter(Analysis.ticker == 'AAPL').all()
print(f'\nQuery 4 - AAPL analyses found: {len(aapl_analyses)}')
for analysis in aapl_analyses:
    print(f'  id={analysis.id}, ticker={analysis.ticker}')

# Close the session.
session.close()


# ==============================================================================
# SECTION 3: UPDATE (Modify a row)
# ==============================================================================
print('\n--- SECTION 3: UPDATE ---')

# Open a new session to find and update the AAPL analysis.
session = SessionLocal()

# Find the AAPL analysis using filter() and first().
aapl_analysis = session.query(Analysis).filter(Analysis.ticker == 'AAPL').first()

# Update the notes field with a new review comment.
aapl_analysis.notes = 'Reviewed — strong buy signal at current RSI'

# Commit the changes to persist the update.
session.commit()

# Close the session.
session.close()

# Open another session to verify the update persisted.
session = SessionLocal()

# Re-fetch the AAPL analysis from the database.
aapl_analysis_updated = session.query(Analysis).filter(Analysis.ticker == 'AAPL').first()

# Print the notes field to confirm the update was successful.
print(f'AAPL analysis notes after update: {aapl_analysis_updated.notes}')

# Close the session.
session.close()


# ==============================================================================
# SECTION 4: DELETE (Remove a row)
# ==============================================================================
print('\n--- SECTION 4: DELETE ---')

# Open a new session to find and delete the TSLA analysis.
session = SessionLocal()

# Find the analysis with ticker == 'TSLA' using filter() and first().
tsla_analysis = session.query(Analysis).filter(Analysis.ticker == 'TSLA').first()

# Delete the TSLA analysis from the session.
session.delete(tsla_analysis)

# Commit the deletion to persist the change.
session.commit()

# Close the session.
session.close()

# Open another session to verify the deletion.
session = SessionLocal()

# Query all remaining analyses.
remaining_analyses = session.query(Analysis).all()

# Print the count of remaining analyses (should be 2: AAPL and BTC).
print(f'Analyses remaining after deletion: {len(remaining_analyses)}')

# Close the session.
session.close()
