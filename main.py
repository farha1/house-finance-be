# main.py
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime, timedelta
import uuid
import uvicorn
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    func,
    extract,
    Boolean,
)  # Added ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship  # Added relationship
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager
from datetime import datetime

# For authentication
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


# --- Database Configuration ---
# Replace with your MySQL connection details
# Format: "mysql+pymysql://user:password@host:port/database_name"
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://dev:dev123@localhost:3306/house_finance"

# Create the SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a SessionLocal class to get a database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# --- SQLAlchemy Database Models ---


class DBUser(Base):
    """
    SQLAlchemy model for the 'users' table.
    """

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


class DBBudget(Base):
    """
    SQLAlchemy model for the 'budgets' table.
    """

    __tablename__ = "budgets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), index=True, nullable=False)
    limit = Column(Float, nullable=False)
    budget_month = Column(Integer, nullable=False)  # Month of the budget (1-12)
    budget_year = Column(Integer, nullable=False)  # Year of the budget (e.g., 2023)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # Define a relationship to Realization.
    # 'backref' creates a 'budget' attribute on Realization instances.
    realizations = relationship("DBRealization", backref="budget", lazy="joined")


class DBRealization(Base):
    """
    SQLAlchemy model for the 'realizations' table.
    """

    __tablename__ = "realizations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    expense_date = Column(Date, nullable=False)
    name = Column(String(200), nullable=False)
    # Define budget_id as a foreign key referencing the 'id' column of the 'budgets' table
    budget_id = Column(String(36), ForeignKey("budgets.id"), nullable=False)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


# Create database tables (if they don't exist)
# This should ideally be handled by migrations in a production environment (e.g., Alembic)
def create_db_tables():
    """Creates all defined tables in the database."""
    Base.metadata.create_all(bind=engine)


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Used to create database tables on startup.
    """
    print("Creating database tables if they don't exist...")
    create_db_tables()
    print("Database tables checked/created.")
    yield
    # Add any cleanup code here if needed for shutdown


# --- FastAPI Application Initialization ---
app = FastAPI(
    title="House Finance Monitor API",
    description="API for managing household budgets and realizations.",
    lifespan=lifespan,  # Use the new lifespan context manager
)

# --- CORS Middleware Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)


# Dependency to get a database session
def get_db():
    """
    Dependency function to provide a database session for each request.
    Ensures the session is closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Authentication Configuration ---
SECRET_KEY = "your-super-secret-key"  # CHANGE THIS IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hashes a plain password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# --- Pydantic Data Models (for request/response validation) ---


# Base model for common fields like ID and creation timestamp
class AppBaseModel(BaseModel):
    """
    Base model to include common fields like a unique ID and creation timestamp.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the record.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the record was created.",
    )


# User models
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserInDB(UserBase):
    hashed_password: str
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# 1. Budget Model
class BudgetBase(BaseModel):
    """
    Base schema for a budget, used for creating new budgets.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the budget (e.g., 'Groceries', 'Utilities').",
    )
    limit: float = Field(
        ..., gt=0, description="Maximum amount allocated for this budget."
    )
    budget_month: Optional[int] = Field(
        ..., ge=1, le=12, description="Month of the expense (1-12)."
    )
    budget_year: Optional[int] = Field(
        ..., ge=2000, le=2100, description="Year of the expense (e.g., 2023)."
    )


class Budget(BudgetBase, AppBaseModel):
    """
    Full budget schema, including ID and creation timestamp, used for responses.
    """

    class Config:
        from_attributes = True  # Pydantic v2: use from_attributes instead of orm_mode


class BudgetUpdate(BaseModel):
    """
    Schema for updating an existing budget. All fields are optional.
    """

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="New name of the budget."
    )
    limit: Optional[float] = Field(
        None, gt=0, description="New maximum amount allocated for this budget."
    )


# 2. Realization Model
class RealizationBase(BaseModel):
    """
    Base schema for a realization (expense), used for creating new realizations.
    """

    expense_date: date = Field(..., description="Date of the expense.")
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Description of the expense (e.g., 'Weekly grocery shopping').",
    )
    budget_id: str = Field(
        ..., description="ID of the budget this realization belongs to."
    )
    amount: float = Field(..., gt=0, description="Amount of the expense.")


class Realization(RealizationBase, AppBaseModel):
    """
    Full realization schema, including ID and creation timestamp, used for responses.
    """

    class Config:
        from_attributes = True  # Pydantic v2: use from_attributes instead of orm_mode


class RealizationUpdate(BaseModel):
    """
    Schema for updating an existing realization. All fields are optional.
    """

    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="New description of the expense.",
    )
    amount: Optional[float] = Field(
        None, gt=0, description="New amount of the expense."
    )


class BudgetTotalRealization(Budget):
    """
    Extends the Budget model to include the total realized amount.
    """

    total_realized: float = Field(
        ..., description="Total amount realized for this budget."
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler for Pydantic validation errors.
    Formats the error messages for better readability.
    """
    errors = []
    for error in exc.errors():
        # Extract the field name from the 'loc' tuple.
        # 'loc' might be like ('body', 'budget_month') or ('query', 'item_id')
        # We want to join parts starting from index 1 to get "budget_month" or "item_id"
        field = ".".join(map(str, error["loc"][1:]))
        message = error["msg"]
        errors.append(f"Field '{field}': {message}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": "Validation failed", "errors": errors},
    )


# --- Authentication Dependencies ---
async def get_user(db: Session, username: str):
    """Retrieves a user from the database by username."""
    return db.query(DBUser).filter(DBUser.username == username).first()


async def authenticate_user(db: Session, username: str, password: str):
    """Authenticates a user by username and password."""
    user = await get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Dependency to get the current authenticated user from the JWT token.
    Raises HTTPException for invalid credentials or token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = await get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# --- API Endpoints ---


# Root endpoint
@app.get("/", summary="Root Endpoint", response_description="A simple welcome message.")
async def read_root():
    """
    Returns a welcome message for the API.
    """
    return {"message": "Welcome to the House Finance Monitor API!"}


# --- User Authentication Endpoints ---


@app.post(
    "/register/",
    response_model=UserBase,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Registers a new user with a unique username and hashed password.
    """
    existing_user = await get_user(db, user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Username already registered"
        )

    hashed_password = get_password_hash(user.password)
    db_user = DBUser(username=user.username, hashed_password=hashed_password)

    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during registration: {e}",
        )

    return db_user


@app.post(
    "/token",
    response_model=Token,
    summary="Login and get an access token",
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    Authenticates a user and returns an access token upon successful login.
    """
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=UserInDB, summary="Get current authenticated user's details", dependencies=[Depends(get_current_user)])
async def read_users_me(current_user: DBUser = Depends(get_current_user)):
    """
    Retrieves details of the currently authenticated and active user.
    """
    return current_user

# --- Budget Endpoints ---


@app.post(
    "/budgets/",
    response_model=Budget,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new budget",
    dependencies=[Depends(get_current_user)],
)
async def create_budget(budget: BudgetBase, db: Session = Depends(get_db)):
    """
    Creates a new budget in the database.
    Validates all required fields and raises appropriate HTTP exceptions.
    """
    try:
        # Validate required fields (will raise ValidationError if missing)
        budget_data = budget.model_dump()

        # Check if a budget with the same name already exists
        existing_budget = (
            db.query(DBBudget)
            .filter(
                DBBudget.name == budget.name,
                DBBudget.budget_month == budget.budget_month,
                DBBudget.budget_year == budget.budget_year,
            )
            .first()
        )
        if existing_budget:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Budget with this name already exists",
            )

        db_budget = DBBudget(**budget_data)
        db.add(db_budget)
        db.commit()
        db.refresh(db_budget)
        return db_budget

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@app.get(
    "/budgets/",
    response_model=List[BudgetTotalRealization],
    summary="Get all budgets",
    dependencies=[Depends(get_current_user)],
)
async def get_budgets(
    db: Session = Depends(get_db), month: int = None, year: int = None
):
    """
    Retrieves a list of all defined budgets from the database.
    """
    now = datetime.now()
    month = month or now.month
    year = year or now.year
    budgets = (
        db.query(DBBudget)
        .filter(
            DBBudget.budget_month == month,
            DBBudget.budget_year == year,
        )
        .all()
    )

    budgets_with_total_realization = []
    for budget in budgets:
        total_realized = (
            db.query(func.sum(DBRealization.amount))
            .filter(DBRealization.budget_id == budget.id)
            .scalar()
        )
        total_realized = total_realized if total_realized is not None else 0.0

        budget_dict = budget.__dict__
        budget_dict["total_realized"] = total_realized
        budgets_with_total_realization.append(BudgetTotalRealization(**budget_dict))

    return budgets


@app.put(
    "/budgets/{budget_id}",
    response_model=Budget,
    summary="Update an existing budget by ID",
    dependencies=[Depends(get_current_user)],
)
async def update_budget(
    budget_id: str, budget_update: BudgetUpdate, db: Session = Depends(get_db)
):
    """
    Updates an existing budget in the database.
    Only provided fields will be updated.
    Raises a 404 error if the budget is not found.
    Raises a 409 error if the updated name/month/year combination already exists.
    """
    budget = db.query(DBBudget).filter(DBBudget.id == budget_id).first()
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found"
        )

    update_data = budget_update.model_dump(
        exclude_unset=True
    )  # Only include fields that were set in the request

    # Check for name/month/year conflict if any of these fields are being updated
    if "name" in update_data:
        new_name = update_data.get("name", budget.name)

        existing_budget_with_new_details = (
            db.query(DBBudget)
            .filter(
                DBBudget.name == new_name,
                DBBudget.budget_month == budget.budget_month,
                DBBudget.budget_year == budget.budget_year,
                DBBudget.id != budget_id,  # Exclude the current budget itself
            )
            .first()
        )
        if existing_budget_with_new_details:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Another budget with this name, month, and year already exists.",
            )

    for key, value in update_data.items():
        setattr(budget, key, value)

    try:
        db.add(budget)  # Add the modified object back to the session
        db.commit()
        db.refresh(budget)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )

    return budget


@app.delete(
    "/budgets/{budget_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a budget by ID",
    dependencies=[Depends(get_current_user)],
)
async def delete_budget(budget_id: str, db: Session = Depends(get_db)):
    """
    Deletes a budget from the database by its unique ID.
    If the budget has associated realizations, they will also be deleted due to CASCADE.
    Raises a 404 error if the budget is not found.
    """
    budget = db.query(DBBudget).filter(DBBudget.id == budget_id).first()
    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found"
        )

    try:
        db.delete(budget)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )
    return {
        "message": "Budget deleted successfully"
    }  # FastAPI expects a response for 204, but content is optional.


# --- Realization Endpoints ---


@app.post(
    "/realizations/",
    response_model=Realization,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new realization",
    dependencies=[Depends(get_current_user)],
)
async def create_realization(
    realization: RealizationBase, db: Session = Depends(get_db)
):
    """
    Creates a new realization (expense) linked to a specific budget in the database.
    Ensures the budget_id exists before creating the realization.
    """
    # Validate if the budget_id exists
    budget_exists = (
        db.query(DBBudget)
        .filter(
            DBBudget.id == realization.budget_id,
            DBBudget.budget_month == realization.expense_date.month,
            DBBudget.budget_year == realization.expense_date.year,
        )
        .first()
    )
    if not budget_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Budget does not exist."
        )

    db_realization = DBRealization(**realization.model_dump())
    try:
        db.add(db_realization)
        db.commit()
        db.refresh(db_realization)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e}",
        )
    return db_realization


@app.get(
    "/realizations/",
    response_model=List[Realization],
    summary="Get all realizations",
    dependencies=[Depends(get_current_user)],
)
async def get_realizations(
    db: Session = Depends(get_db), month: int = None, year: int = None
):
    """
    Retrieves a list of all recorded realizations (expenses) from the database.
    """
    query = db.query(DBRealization)
    if month is not None:
        query = query.filter(extract("month", DBRealization.expense_date) == month)
    if year is not None:
        query = query.filter(extract("year", DBRealization.expense_date) == year)

    realizations = query.all()
    return realizations


@app.put(
    "/realizations/{realization_id}",
    response_model=Realization,
    summary="Update an existing realization by ID",
    dependencies=[Depends(get_current_user)],
)
async def update_realization(
    realization_id: str,
    realization_update: RealizationUpdate,
    db: Session = Depends(get_db),
):
    """
    Updates an existing realization in the database.
    Only provided fields will be updated.
    Raises a 404 error if the realization is not found.
    Raises a 400 error if the new budget_id does not exist.
    """
    realization = (
        db.query(DBRealization).filter(DBRealization.id == realization_id).first()
    )
    if not realization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Realization not found"
        )

    update_data = realization_update.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(realization, key, value)

    try:
        db.add(realization)  # Add the modified object back to the session
        db.commit()
        db.refresh(realization)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )

    return realization


@app.delete(
    "/realizations/{realization_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a realization by ID",
    dependencies=[Depends(get_current_user)],
)
async def delete_realization(realization_id: str, db: Session = Depends(get_db)):
    """
    Deletes a realization from the database by its unique ID.
    Raises a 404 error if the realization is not found.
    """
    realization = (
        db.query(DBRealization).filter(DBRealization.id == realization_id).first()
    )
    if not realization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Realization not found"
        )

    try:
        db.delete(realization)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )
    return {"message": "Realization deleted successfully"}


# To run this application:
# 1. Save the code as `main.py`
# 2. Install FastAPI, Uvicorn, SQLAlchemy, and PyMySQL: `pip install fastapi uvicorn sqlalchemy pymysql`
# 3. Ensure your MySQL server is running and you have created the database specified in SQLALCHEMY_DATABASE_URL.
# 4. Run the application using Uvicorn: `uvicorn main:app --reload`
# 5. Access the API documentation at `http://127.0.0.1:8000/docs`
